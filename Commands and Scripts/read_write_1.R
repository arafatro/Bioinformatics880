# just check the number of Lysine in our data bank:
rm(list = ls())

Methylation = read.csv('C:\\Users/idehz/Desktop/Methylation/Methylation_40.csv',header = FALSE)
#Methylation = read.csv('C:\\Users/idehz/Desktop/Collaborations/PTM_new/Methylation/Methylation.elm')
dim(Methylation)
#Methylation[1:5,1]
Methylation[2,]
#check uniques

Methylation_uniq = Methylation[!duplicated(Methylation[,1]), ]
dim(Methylation_uniq)
head(Methylation_uniq)
Methylation_uniq[1:10,1:2]

#write.csv(Methylation_uniq,file = 'C:\\Users/idehz/Desktop/Collaborations/PTM_new/Methylation/Methylation_uniq.csv')
#write.csv(Methylation,file = 'C:\\Users/idehz/Desktop/Collaborations/PTM_new/Methylation/Methylation.csv')
#write.csv(Methylation_uniq,file = 'C:\\Users/idehz/Desktop/Methylation/Methylation_uniq.csv')
#x = read.csv('C:\\Users/idehz/Desktop/Methylation/Methylation_uniq.csv')
#colnames(x)
#x[1:5,5]
# saving the unique list of proteins in methylation as .csv:
#write.csv(Methylation_uniq,file = '~/Downloads/Protein_database/Methylation/Methylation_uniq2.csv')
length(!is.na(Methylation[,2]))

for(i in 1:length(Methylation[,1]) )
{
 
  write(c(paste0(Methylation[i,1]),as.character(Methylation[i,2])),file = 'C:\\Users/idehz/Desktop/Methylation/Methylation_40.fasta',append = TRUE)
}

rm(list = ls())

Methylation = read.csv('C:\\Users/idehz/Desktop/Methylation/Methylation_40_name.csv',header = FALSE)

for(m in 1:length(Methylation[,1]))
{
  file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq40/',as.character(Methylation[m,1]),'.seq'))
  writeLines(c(paste0('>',Methylation[m,1]),as.character(Methylation[m,2])),sep = "\n",file_name)
  close(file_name)
}

