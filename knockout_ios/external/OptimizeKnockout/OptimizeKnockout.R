#!/usr/bin/Rscript
rm(list=ls())

#### Command line arguments handling ####
getArgs <- function(verbose=FALSE, defaults=NULL) {
  myargs <- gsub("^--","",commandArgs(TRUE))
  setopts <- !grepl("=",myargs)
  if(any(setopts))
    myargs[setopts] <- paste(myargs[setopts],"=notset",sep="")
  myargs.list <- strsplit(myargs,"=")
  myargs <- lapply(myargs.list,"[[",2 )
  names(myargs) <- lapply(myargs.list, "[[", 1)
  
  ## logicals
  if(any(setopts))
    myargs[setopts] <- TRUE
  
  ## defaults
  if(!is.null(defaults)) {
    defs.needed <- setdiff(names(defaults), names(myargs))
    if(length(defs.needed)) {
      myargs[ defs.needed ] <- defaults[ defs.needed ]
    }
  }
  
  ## verbage
  if(verbose) {
    cat("read",length(myargs),"named args:\n")
    print(myargs)
  }
  myargs
}

args <- getArgs()
if(is.null(args$inputData)){
  stop("please specify at least input data file name")
}

if(args$inputData == "Envpermit.csv") {

args <- getArgs(defaults = list(koActivities='c("T02","T06","T10")',
                                 usefulFeatures = 'c("Resource","X.case..channel","X.case..department","X.case..group","X.case..responsible",
                                                   "org.group")',
                                disallowed_permutation= 'c("T10","T06","T02","T10","T02","T06")',
                                learner="rf",sampling_method="under",n=1,retrain=TRUE
                                  ))
}

if(args$inputData == "Bondora.csv") {
  
  args <- getArgs(defaults = list(koActivities='c("IdCancellation", "PostFundingCancellation", "CreditDecision")',
                                  usefulFeatures = 'c("Age","Gender","Country","NewCreditCustomer","language_code","education_id",
                                                     "marital_status_id","nr_of_dependants","employment_status_id",
                                                     "Employment_Duration_Current_Employer","work_experience","occupation_area",
                                                     "home_ownership_type_id",
                                                     "income_from_principal_employer","income_total","TotalLiabilitiesBeforeLoan",
                                                     "TotalMonthlyLiabilities","DebtToIncome",
                                                     "AppliedAmountToIncome","LiabilitiesToIncome","NoOfPreviousApplications",
                                                     "AmountOfPreviousApplications",
                                                     "AppliedAmount","Interest","LoanDuration","UseOfLoan","ApplicationType")',
                                  numFeatures = 'c("AppliedAmount","Interest","LoanDuration","nr_of_dependants","income_from_principal_employer",
                                                  "income_total",
                                                  "TotalLiabilitiesBeforeLoan","TotalMonthlyLiabilities","DebtToIncome",
                                                  "AppliedAmountToIncome","LiabilitiesToIncome","NoOfPreviousApplications",
                                                  "AmountOfPreviousApplications")',
                                  learner="svm",sampling_method="under",n=1,retrain=TRUE
  ))
}

koActivities = eval(parse(text=args$koActivities))
usefulFeatures = eval(parse(text=args$usefulFeatures))
usefulFeatures = c(usefulFeatures,koActivities,paste(koActivities,"_time",sep = ""))
if(!is.null(args$numFeatures)) {
  numFeatures = eval(parse(text=args$numFeatures))
} else {
  numFeatures=c()
}
if(!is.null(args$disallowed_permutation))  {disallowed_permutation = eval(parse(text=args$disallowed_permutation))
  disallowed_permutation = matrix(disallowed_permutation,ncol = length(koActivities),byrow = TRUE)
} else {
  disallowed_permutation = c()
  }
learner = args$learner
sampling_method = args$sampling_method
inputData = args$inputData
cleanedFileName = substring(inputData,1,nchar(inputData) - 4)
n = as.integer(args$n)
retrain = args$retrain

#### Main part ####
mainDir <- getwd()
outputDir <- "output"
learnerDir <- "learners"
if(!dir.exists(file.path(mainDir, learnerDir))) dir.create(file.path(mainDir, learnerDir))
if(!dir.exists(file.path(mainDir, outputDir))) dir.create(file.path(mainDir, outputDir))

source('functions_overprocessing.R')

computeOverprocessing(inputData,  koActivities, usefulFeatures, numFeatures,disallowed_permutation,n)


#### Results ####
setwd(outputDir)
filenames <- list.files()[grep(paste("output_" ,cleanedFileName,"_",learner,"_", sampling_method, "_(?=.*\\.csv)",sep = ""), list.files(), perl=TRUE)]
print(filenames)
dat = c()

for (i in 1:length(filenames)) {
  foo = read.table(filenames[i],sep=",",header=TRUE)
  dat = rbind(dat,foo)
  }

tt = matrix(0,nrow = 4,ncol = length(unique(dat$nr_checks_rand)))
rownames(tt) = c("count_checks_our","count_checks_base","count_checks_rand","num_checks_minimal")
tt[1,] = round(table(dat$nr_checks_our)/length(filenames))
tt[2,] = round(table(dat$nr_checks_Wil)/length(filenames))
tt[3,] = round(table(dat$nr_checks_rand)/length(filenames))
tt[4,1] = round(sum(dat$minimum_check_number == 1)/length(filenames))
tt[4,2] = 0
tt[4,3] = round(sum(dat$minimum_check_number == 3)/length(filenames))
colnames(tt) = names(table(dat$nr_checks_rand))

fileSummary = paste("output_" ,cleanedFileName,"_",learner,"_", sampling_method, "_summary.txt",sep = "")
write (
  "***** STATISTICS ******* ", file = fileSummary, append = FALSE, sep = ""
)
cat (
  "Minimal possible number of checks: ", file = fileSummary, append = TRUE, sep = ""
)
cat(
  round(sum(dat$minimum_check_number)/length(filenames)), file = fileSummary, append = TRUE, sep = "\t"
)
cat (
  c("\n", "Average number of checks one would do if they follow our ordering: "), file = fileSummary, append = TRUE, sep = ""
)
cat(
  round(sum(dat$nr_checks_our)/length(filenames)), file = fileSummary, append = TRUE, sep = "\t"
)
cat (
  "\n", "Average number of checks with the baseline ordering (constant prob's): ", file = fileSummary, append = TRUE, sep = ""
)
cat(
  round(sum(dat$nr_checks_Wil)/length(filenames)), file = fileSummary, append = TRUE, sep = "\t"
)
cat (
  "\n", "Average number of checks with the random ordering: ", file = fileSummary, append = TRUE, sep = ""
)
cat(
  c(round(sum(dat$nr_checks_rand)/length(filenames)), "\n"), file = fileSummary, append = TRUE, sep = "\t"
)
cat (
  c("\n","Distribution of the number of checks performed: ","\n"), file = fileSummary, append = TRUE, sep = ""
)
write.table(
  tt, file = fileSummary, append = TRUE, sep = "\t"
)