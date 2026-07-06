
setwd(this.path::here())

biometrics = read.csv('../source/biometrics.csv')
claims = read.csv('../source/claims.csv')
admin = read.csv('../source/firm_admin.csv')
marathon = read.csv('../source/marathon.csv')
survey = read.csv('../source/online_surveys.csv')
participation = read.csv('../source/participation.csv')
sources = list(
  participation,
  claims,
  biometrics,
  survey,
  admin,
  marathon
)
control_vars = c(
  'male',
  'age50',
  'age37_49',
  'white',
  'salaryQ1',
  'salaryQ2',
  'salaryQ3',
  'salaryQ4',
  'faculty',
  'AP',
  'everscreen_0716',
  'active_0716',
  'active_try_0716',
  'cursmk_0716',
  'othersmk_0716',
  'formsmk_0716',
  'drink_0716',
  'drinkhvy_0716',
  'chronic_0716',
  'health1_0716',
  'health2_0716',
  'problems_0716',
  'energy_0716',
  'ehealth_0716',
  'overweight_0716',
  'badhealth_0716',
  'sedentary_0716',
  'druguse_0716',
  'physician_0716',
  'hospital_0716',
  'sickdays_0716',
  'hrsworked50_0716',
  'jobsatisf1_0716',
  'jobsatisf2_0716',
  'mgmtsafety_0716',
  'spendHosp_0715_0716',
  'spendOff_0715_0716',
  'spendRx_0715_0716',
  'spend_0715_0716',
  'nonzero_spend_0715_0716',
  'sickleave_0815_0716',
  'gym_0815_0716',
  'marathon_2014_2016'
)
outcome_vars = c(
  'nonzero_spend_0816_0717',
  'spendHosp_0816_0717',
  'spendOff_0816_0717',
  'spendRx_0816_0717',
  'spend_0816_0717',
  'sickleave_0816_0717',
  'gym_0816_0717',
  'marathon_2017'
)
combined = do.call(cbind, sources)
nonmissing_rows = !is.na(combined[["spend_0816_0717"]])
combined = combined[nonmissing_rows,]
dim(combined)
combined = combined[,colSums(is.na(combined))<nrow(combined)]
combined_missing = 1*is.na(combined)

for(i in 1:ncol(combined)) {
  col_mean = mean(combined[,i],na.rm=TRUE)
  missing = is.na(combined[,i])
  combined[missing,i] = col_mean
}

controls = combined[,control_vars]
controls_missing = combined_missing[,control_vars]
outcomes = combined[,outcome_vars]
treatment = combined['treat']

write.csv(combined,'combined.csv',row.names=FALSE)
write.csv(combined_missing,'combined_missing.csv',row.names=FALSE)
write.csv(controls,'controls.csv',row.names=FALSE)
write.csv(controls_missing,'controls_missing.csv',row.names=FALSE)
write.csv(outcomes,'outcomes.csv',row.names=FALSE)
write.csv(treatment,'treatment.csv',row.names=FALSE)
