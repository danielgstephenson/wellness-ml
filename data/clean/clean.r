
setwd(this.path::here())

biometrics = read.csv('../source/biometrics.csv')
claims = read.csv('../source/claims.csv')
admin = read.csv('../source/firm_admin.csv')
marathon = read.csv('../source/marathon.csv')
survey = read.csv('../source/online_surveys.csv')
participation = read.csv('../source/participation.csv')
sources = list(
  participation,
  claims
)
combined = do.call(cbind, sources)
dim(combined)
clean = combined[,sapply(combined,is.numeric)]
dim(clean)
clean = combined[complete.cases(combined),]
# dim(clean)


write.csv(clean,'clean.csv',row.names=FALSE)
