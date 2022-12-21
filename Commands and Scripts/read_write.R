# just check the number of Lysine in our data bank:
rm(list = ls())
Methylation = read.csv('C:\\Users/idehz/Desktop/Methylation/Methylation_40_name.csv',header=FALSE)

length(unique(Methylation[,1]))

#Methylation_uniq = NULL
#for(i in unique(Methylation[,1]))
#{
#    indx = which(Methylation[,1] == i)[1]
#    Methylation_uniq = rbind(Methylation_uniq,Methylation[indx,])
#}
#Succulynation_uniq[1:10,1]
# No need for this one. It is already sorted as I follow the levels:
#Succulynation_uniq1 = Succulynation[as.character(sort(as.numeric(rownames(Succulynation_uniq)))),]
###############################################
# Saving the unique list: Succulynation_uni
#write.csv(Methylation_uniq,file = '~/Downloads/Protein_database/Methylation/Methylation_uniq.csv')
# writing each protein in a single file and adding the name of the seqence there.
for(i in 1:length(Methylation[,1]))
{
    file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq_filter/',as.character(Methylation[i,1]),'.seq'))
    writeLines(c(paste0('>',Methylation[i,1]),as.character(Methylation[i,2])),sep = "\n",file_name)
    close(file_name)
}

