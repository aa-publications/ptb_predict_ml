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
### Abin Arbraham 
### Dec 7, 2018
### R >3.4.3 


.libPaths("~/R/rlib-3.4.3/")
library(icd)
library(dplyr)
library(readr)
library(argparse)

sessionInfo()
options(warn=-1)

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


# load feature importance file preserving icd9 columns as character
feat_import_df <- read_delim(feat_file, "\t", escape_double = FALSE, col_types = cols(feature = col_character()), trim_ws = TRUE)
print("Input data:")
head(feat_import_df)

# convert to icd-9 decsription 
translated = explain_table(as.icd9(feat_import_df$feature), short_code=FALSE)
translated = translated %>% select(c(-billable, -is_major, -code, -long_desc, -valid_icd10,-three_digit))

# add icd descriptions 
concat_df = bind_cols(feat_import_df, translated)
print("Feature importance data combined with icd9 descriptions:")
head(concat_df)

# if NA, replace with feature 
concat_df = concat_df %>% mutate(short_desc = if_else(is.na(short_desc), feature, short_desc))
concat_df = concat_df %>% mutate(major = if_else(is.na(major), feature, as.character(major)))
concat_df = concat_df %>% mutate(sub_chapter = if_else(is.na(sub_chapter), feature, as.character(sub_chapter)))
concat_df = concat_df %>% mutate(chapter = if_else(is.na(chapter), feature, as.character(chapter)))
         
                      
# for not valid_icd9 flag, add a ? to the desciprtion
concat_df = concat_df %>% mutate(short_desc_w_valid = if_else(valid_icd9 == FALSE, paste("?", short_desc, sep=""), paste("", short_desc, sep="")))
sprintf("Number of features that did have a matching icd-9 description: %s", sum(concat_df$valid_icd9==FALSE) )

# write descriptions 
write_delim(concat_df, path=FEAT_DESCRIPTION_FILE, delim="\t")
sprintf("Wrote features with icd9 descriptions to:")
sprintf(FEAT_DESCRIPTION_FILE)

print("Done!")
