### Convert ICD9 to their descriptions 
### 
###   requires that icd codes are icd9! 
###
###
###
### INPUTS: 
###       - tsv file with at least one columns with header 'feature'
###       - full path to output directory to write output file to 
###
### OUTPUTS: 
###       - tsv file with above columns and description of icd-9 codes appended 
###
### EXCEPTION HANDLING: 
###       - if no description can be found, the code itself will be used as the description
###
### Abin Arbraham 
### Dec 7, 2018
### R >3.4.3 

### UPDATED on 2019-03-29 08:16:55 to map CPT to description I obtained from Adi.


.libPaths("~/R/rlib-3.4.3/")
library(icd)
library(dplyr)
library(readr)
library(argparse)

sessionInfo()
options(warn=-1)

# DEPENDENCIES FILE: 
CPT_CODES = "/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/combined_cohorts_07_12_2018/full_cpt_codes/uniq_CPT_codes.txt"
CPT_CODE_DESCRIPTIONS ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/cpt_codes_manual/athena_cpt_labels.tsv"
CPT_CODE_DESCRIPTIONS ="/dors/capra_lab/users/abraha1/projects/PTB_phenotyping/data/cpt_codes_manual/CPT4.MapOffline.Feb2019.cache.csv"


### SET UP ARGPARSE
parser <- ArgumentParser()
parser$add_argument("-f", "--feat_file", type="character", dest="feat_file", default="0",
                    help="full path to feature importance file")

parser$add_argument("-l", "--data_label", type="character", dest="dlabel", default="",
                    help="short description of this dataset, defaults to nothing")


parser$add_argument("-o", "--out_dir", type="character", dest="outdir", default=getwd(),
                    help="output dir path for snp")

args <- parser$parse_args()
feat_file = args$feat_file
dlabel = args$dlabel 
output_dir= args$outdir

# append prefix to input file name 
output_file = paste(sprintf("descrip_%s", dlabel), basename(feat_file), sep="-")
FEAT_DESCRIPTION_FILE = file.path(output_dir, output_file)

# load all cpt codes 
cpt_descrip = read_delim(CPT_CODE_DESCRIPTIONS, "\t")
cpt_descrip =  cpt_descrip %>% select(c(CODE, NAME)) %>% rename(feature=CODE)
cpt_codes = cpt_descrip %>% select(feature)


# load feature importance file preserving icd9 columns as character
feat_import_df <- read_delim(feat_file, "\t", escape_double = FALSE, col_types = cols(feature = col_character()), trim_ws = TRUE)
print("Input data:")
head(feat_import_df)

# isolate icd and cpt codes 
no_cpt_feat_import_df = feat_import_df %>% filter( !(feature %in% cpt_codes$feature))
cpt_feat_import_df = feat_import_df %>% filter(feature %in% cpt_codes$feature)

# convert to icd-9 decsription 
translated = explain_table(as.icd9(no_cpt_feat_import_df$feature), short_code=FALSE)
translated = translated %>% select(c(-billable, -is_major, -code, -long_desc, -valid_icd10,-three_digit))
translated$icd = no_cpt_feat_import_df$feature
translated$importance = no_cpt_feat_import_df$importance
translated = translated %>% rename(feature=icd)

# convert to CPT descriptions
translated_cpt = left_join(cpt_feat_import_df, cpt_descrip,by="feature") %>% rename(short_desc=NAME)

# combind icd and cpt 
if ("importance" %in% colnames(translated) & "importance" %in% colnames(translated_cpt))
{
    combine_df = bind_rows(translated, translated_cpt) %>% arrange(desc(importance))
    final_df = combine_df %>% select(feature, importance, short_desc, major)
} else { 
    combine_df = bind_rows(translated, translated_cpt)
    final_df = combine_df %>% select(feature, short_desc, major)
}

# add icd descriptions 
print("Feature importance data combined with cpt and  icd9 descriptions:")
head(final_df)

# if NA, replace with feature 
final_df = final_df %>% mutate(short_desc = if_else(is.na(short_desc), feature, short_desc))
final_df = final_df %>% mutate(major = if_else(is.na(major), feature, as.character(major)))
    
# write descriptions 
write_delim(final_df, path=FEAT_DESCRIPTION_FILE, delim="\t")
sprintf("Wrote features with icd9 descriptions to:")
sprintf(FEAT_DESCRIPTION_FILE)

print("Done!")
