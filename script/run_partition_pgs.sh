#!/bin/bash

PLINK=./plink
TABIX=tabix
BEDTOOLS=bedtools

BETA_NAME="beta"
SE_NAME="se"
K=50
WINDOW=1000
TEMPDIR="temp"

progname=$(basename $0)

function print_help () {
    cat << EOF

Usage: $progname --ld=LD_FILE --gwas=GWAS_BED_FILE --geno=GENO_HDR --annot=ANNOT_BED_FILE --out=OUT_FILE [--window=WINDOW] [--beta=BETA_NAME] [--se=SE_NAME] [--temp=TEMPDIR] [--numsvd=K] [--plink=PLINK] [--bedtools=BEDTOOLS] [--tabix=TABIX]

[Required]

LD_FILE		: LD BED file
GWAS_BED_FILE	: GWAS BED file (bgzipped with tabix)
GENO_HDR	: genotype (plink fileset) header
ANNOT_BED_FILE	: annotation BED file (bgzipped with tabix)
OUT_FILE	: output file

[Optional]

BETA_NAME	: GWAS beta column name (default: "beta")
SE_NAME		: standard error column (default: "se")
WINDOW		: window size to extend the annotated regions (default: $WINDOW)
K		: LD complexity, number of SVs (default: $K)
TEMPDIR		: temporary directory (default: "temp")

[Binary files]

PLINK		: plink binary path (default: "$PLINK")
TABIX		: tabix binary path (default: "$TABIX")
BEDTOOLS	: bedtools binary path (default: "$BEDTOOLS")

EOF
}

if [ $# -lt 1 ]; then
    print_help
    exit 1
fi

for i in "$@"
do
    case $i in
	--ld=*)
	    LD_FILE="${i#*=}"
	    shift
	    ;;
	--gwas=*)
	    GWAS_BED_FILE="${i#*=}"
	    shift
	    ;;
	--geno=*)
	    GENO_HDR="${i#*=}"
	    shift
	    ;;
	--annot=*)
	    ANNOT_BED_FILE="${i#*=}"
	    shift
	    ;;
	--out=*)
	    OUT_FILE="${i#*=}"
	    shift
	    ;;
	--numsvd=*)
	    K="${i#*=}"
	    shift
	    ;;
	--beta=*)
	    BETA_NAME="${i#*=}"
	    shift
	    ;;
	--se=*)
	    SE_NAME="${i#*=}"
	    shift
	    ;;
	--window=*)
	    WINDOW="${i#*=}"
	    shift
	    ;;
	--temp=*)
	    TEMPDIR="${i#*=}"
	    shift
	    ;;
	--plink=*)
	    PLINK="${i#*=}"
	    shift
	    ;;
	--tabix=*)
	    TABIX="${i#*=}"
	    shift
	    ;;
	--bedtools=*)
	    BEDTOOLS="${i#*=}"
	    shift
	    ;;
	-h|--help)
	    print_help
	    exit 1
	    ;;
	*)
	    echo "Unknown options"
	    ;;
    esac
done

ZERO=0
[ -z ${LD_FILE+ZERO} ] && exit 1
[ -z ${GWAS_BED_FILE+ZERO} ] && exit 1
[ -z ${ANNOT_BED_FILE+ZERO} ] && exit 1
[ -z ${OUT_FILE+ZERO} ] && exit 1

# Make sure that we don't overwrite
mkdir -p $TEMPDIR
TEMPDIR=$(mktemp -d "${TEMPDIR}.XXXXXXXX")
if [ -d $TEMPDIR ] ; then
    echo "TEMPDIR $TEMPDIR already exists"
    rm -r -f $TEMPDIR/*
fi

nLD=$(cat $LD_FILE | tail -n+2 | wc -l)

for((ld_idx=1; ld_idx<=$nLD; ++ld_idx)); do

    mkdir -p $TEMPDIR/$ld_idx

    interval=$(cat $LD_FILE | tail -n+2 | head -n $ld_idx | tail -n1 | awk '{ print $1 ":" $2 "-" $3 }')
    chr=$(cat $LD_FILE | tail -n+2 | head -n $ld_idx | tail -n1 | awk '{ print $1 }')
    lb=$(cat $LD_FILE | tail -n+2 | head -n $ld_idx | tail -n1 | awk '{ print $2 }')
    ub=$(cat $LD_FILE | tail -n+2 | head -n $ld_idx | tail -n1 | awk '{ print $3 }')

    echo "Evaluating $interval [$ld_idx / $nLD] ..."

    plink_hdr=$GENO_HDR/$chr

    echo "Finding PLINK $plink_hdr ..."

    [ -f $plink_hdr.bed ] || continue

    ###################################################
    # 1. Intersection between GWAS and annotation BED #
    ###################################################

    gwas_ld=$TEMPDIR/$ld_idx/gwas.bed.gz
    annot_ld=$TEMPDIR/$ld_idx/annot.bed.gz
    gwas_ld_dir=$TEMPDIR/$ld_idx/stratified

   
    $TABIX -h $GWAS_BED_FILE $interval | sort -k1,1 -k2,2n | bgzip -c > $gwas_ld
    # Check empty GWAS in this LD block
    n=$(cat $gwas_ld | gzip -d | wc -l)

    if [ $n -lt 2 ]; then
	interval2=$(echo $interval | sed 's/chr//g')
	$TABIX -h $GWAS_BED_FILE $interval2 | awk 'NR == 1 { print } NR > 1 { print "chr" $0 }' | sort -k1,1 -k2,2n | bgzip -c > $gwas_ld
    fi

    # Check empty GWAS in this LD block
    n=$(cat $gwas_ld | gzip -d | wc -l)
    [ $n -gt 2 ] || continue

    $TABIX -p bed $gwas_ld

    # Check empty annotation in this LD block
    na=$($TABIX $ANNOT_BED_FILE $interval | wc -l)
    [ $na -gt 1 ] || continue
    
    echo "Parsing annotations ..."
    annot_names=$($TABIX $ANNOT_BED_FILE $interval | awk -F'\t' '{ annot[$NF]++ } END { for(a in annot) print a }')
    echo "$annot_names"
    
    for aa in $annot_names; do

	gwas_ld_aa=$gwas_ld_dir/$aa.bed.gz
	mkdir -p $(dirname $gwas_ld_aa)

	cat $GWAS_BED_FILE | bgzip -d | head -n1 | bgzip -c > $gwas_ld_aa

	$TABIX $ANNOT_BED_FILE $interval | \
	    awk -F '\t' -v W=$WINDOW -v A=$aa \
		'BEGIN {}
function max(a,b) {
    return a > b ? a : b
}
$4 == A {
    print $1 FS max($2 - W, 0) FS ($3 + W) FS $4
}' | \
	    $BEDTOOLS intersect -a stdin -b $gwas_ld -wa -wb | \
	    cut -f 5- | sort -k1,1 -k2,2n | \
	    bgzip -c >> $gwas_ld_aa

	# echo "Generating $gwas_ld_aa ..."
	$TABIX -p bed $gwas_ld_aa
    done

    #########################################
    # 2. Take a subset of the plink fileset #
    #########################################

    plink_ld=$TEMPDIR/$ld_idx/plink

    $PLINK --bfile $plink_hdr --make-bed \
	   --chr $chr --from-bp $lb --to-bp $ub \
	   --out $plink_ld

    #################################
    # 3. Calculate polygenic scores #
    #################################

    pgs_ld=$TEMPDIR/$ld_idx.pgs.gz

    ./pgs_calc_ld.R $ld_idx \
		    $gwas_ld \
		    $gwas_ld_dir \
		    $plink_ld \
		    $BETA_NAME \
		    $SE_NAME \
		    $K \
		    $pgs_ld || exit 1

    rm -r $TEMPDIR/$ld_idx/
done

################################################################
# 4. Combine
./pgs_combine.R $TEMPDIR/ $OUT_FILE || exit 1

[ -d $TEMPDIR ] && rm -r $TEMPDIR

echo "Done"
exit 0
