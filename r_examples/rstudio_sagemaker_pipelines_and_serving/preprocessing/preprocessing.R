library(readr)
library(dplyr)
library(ggplot2)
library(forcats)

input_dir <- "/opt/ml/processing/input/"
output_dir <- "/opt/ml/processing/output/"
#dir.create(output_dir, showWarnings = FALSE)

filename <- Sys.glob(paste(input_dir, "*.csv", sep=""))
abalone <- read_csv(filename)

abalone <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)
abalone <- abalone %>% select(rings:infant, length:shell_weight)


abalone_train <- abalone %>%
  sample_frac(size = 0.7)
abalone <- anti_join(abalone, abalone_train)
abalone_test <- abalone %>%
  sample_frac(size = 0.5)
abalone_valid <- anti_join(abalone, abalone_test)


write_csv(abalone_train, paste0(output_dir,'train/abalone_train.csv'))

write_csv(abalone_valid, paste0(output_dir,'valid/abalone_valid.csv'))
write_csv(abalone_test, paste0(output_dir,'test/abalone_test.csv'))

# Remove target from test
# write_csv(abalone_test[-1], paste0(output_dir,'abalone_test.csv'), col_names = FALSE)

