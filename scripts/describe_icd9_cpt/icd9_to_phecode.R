#!/usr/bin/env Rscript
### This script will convert ICD-9 code count to PheCodes.  
### 
###     INPUTS: 
###         * input file should be a tab seperated file with the following columns: 'GRID','ICD','Date','concat_row'
###                - file should have one row per GRID-ICD-DATE combo 
###
###
### Abin Abraham
### Nov 23, 2018 
### R/3.3.3



### SET DIRECTORY AND LOAD PACKAGES 
.libPaths("~/R/rlib-3.3.3/")
library(PheWAS)
library(argparse)
library(readr)

### SET UP ARGPARSE
parser <- ArgumentParser()
parser$add_argument("-f", "--icd_file", type="character", dest="icd_file", default="0",
                    help="full path to icd file")

parser$add_argument("-l", "--dataset_label", type="character", dest="dlabel", default='',
                    help="label for the dataset that will be appened to output file name")

parser$add_argument("-o", "--out_dir", type="character", dest="outdir", default=getwd(),
                    help="output dir path for snp")

args <- parser$parse_args()
icd_file = args$icd_file
dlabel = args$dlabel
output_dir= args$outdir

# output files that will be written
phe_output_file = file.path(output_dir, sprintf("phecodes_%s_feat_mat.tsv", dlabel))
phe_with_exclusion_output_file = file.path(output_dir, sprintf("phecodes_%s_with_exclusion_feat_mat.tsv", dlabel))


# deafults for debugging 
#  
# icd_file = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_dataset/full_ICD9_cohort.tsv"
# dlabel =""
# output_dir = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/scripts/rand_forest_ptb_classification/out"


icd_df = read.csv(icd_file, sep="\t", col.names = c('GRID','ICD','Date','concat_row'),
                            colClasses=c("character","character","character","character"))


# ICD9 CODES THAT CODE FOR PRETERM, TERM, OR POST-TERM
PRETERM_ICD9_CODES = c('644.2', '644.20', '644.21')
TERM_ICD9_CODES = c('650', '645.1', '645.10', '645.11', '645.13', '649.8', '649.81', '649.82')
POSTTERM_ICD9_CODES = c('645.2', '645.20', '645.21', '645.23', '645.00', '645.01', '645.03')
ALL_DELIVERY_CODES = c(PRETERM_ICD9_CODES, TERM_ICD9_CODES, POSTTERM_ICD9_CODES)
print("Delivery based ICD-9 codes have been excluded...")

# remove delivery based icd codes 
filtered_icd_df = icd_df %>% filter(!ICD %in% ALL_DELIVERY_CODES)

# get count of ICDs per person
icd_counts = filtered_icd_df %>% group_by(GRID, ICD) %>% count() %>% rename(count = n)
icd_counts = as.data.frame(icd_counts)
orig_phe_table = createPhewasTable(icd_counts, min.code.count = 2)

# converts NA and FALSE to 0, TRUE to 1 
numeric_no_excl_phe = orig_phe_table %>% replace(is.na(.), 0)

# converts NA to -1, FALSE to 0, and TRUE to 1
numeric_excl_phe = orig_phe_table %>% replace(is.na(.), -1)

# path=file.path(output_dir, "excl_phe.tsv")
write_delim(numeric_no_excl_phe, phe_output_file, delim = "\t", na = "NA", append = FALSE)
write_delim(numeric_excl_phe, phe_with_exclusion_output_file, delim = "\t", na = "NA", append = FALSE)



sprintf("Output files are written to:\n%s", output_dir)
print("Done converting file to PheCodes!")
