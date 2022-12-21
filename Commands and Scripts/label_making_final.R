# Producing the labels for our protein sequences:

rm(list = ls())
#load the data:
PTM_But =read.csv('C:\\Users/idehz/Desktop/Methylation/Methylation.csv')
len = length(unique(PTM_But[,1]))
#PTM_But[1:10,]
label=matrix(0,nrow = len,ncol = 3)
dim(label)
count =1
#going through all the unique protiens:
length(unique(PTM_But[,1]))
s = unique(PTM_But[,1])
#PTM_But[1,4]
s[1:10]
#i = s[1]

#PTM_But[1:10,2]

for(i in s)
 {
  #################
  #############
  indx = which(PTM_But[,1] == i)[1]
  #indx
  #print(indx)
  #split them to work as character with them:
  seq_in = strsplit(as.character(PTM_But[indx,4]),split='')
  #seq_in
  #saving the output
  label[count,1] = as.character(PTM_But[indx,1])
  label[count,2] = as.character(PTM_But[indx,4])
  #label[1,]
  # remember for your loop, you shoud lave a sequence. i in seq_in does not working! instead: i in 1:seq[[1]]
  #    count =1
  
  flag=0
  seq_out=NULL
  s1=NULL
  for(j in 1:length(seq_in[[1]]))
  {
    #print(j)
    if(as.character(seq_in[[1]][j]) != as.character('K'))
    {
      seq_out=c(seq_out,as.character('2'))
      #print(as.character(i) == 'K')
    }
    if(as.character(seq_in[[1]][j]) == as.character('K'))
    {
      
      #seq_out=c(seq_out,'1')
      #Check1= PTM_But[PTM_But[,1]==levels(PTM_But[,1])[i],]
      Check1= PTM_But[PTM_But[,1]==i,]
      flag=0
      for(n in 1:length(Check1[,1]))
      {
        if(as.character(j)==Check1[n,3])
        {
          flag=1
        }
      }
      if(flag==1)
      {
        seq_out=c(seq_out,'1')
      }
      else
      {
        seq_out=c(seq_out,'0')
      }
      flag=0
    }
  }
  for(i in seq_out)
    s1=paste0(s1,i,sep='')
  label[count,3]=as.character(s1)
  s1=NULL
  count = count +1
  seq_out=NULL
  seq_in=NULL
  
  ########
  #######
}
dim(label)
#label[1:10,1]
write.csv(label,file = 'C:\\Users/idehz/Desktop/Methylation/label_seq_name.csv')
#label[,1]
for(m in 1:length(label[,1]))
{
    file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq/',as.character(label[m,1]),'.lab'))
    writeLines(as.character(label[m,3]),sep = "\n",file_name)
    close(file_name)
}
for(m in 1:length(label[,1]))
{
    file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq/',as.character(label[m,1]),'.name.lab'))
    writeLines(c(paste0('>',label[m,1]),as.character(label[m,3])),sep = "\n",file_name)
    close(file_name)
}
for(m in 1:length(label[,1]))
{
    file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq/',as.character(label[m,1]),'.name.seq.lab'))
    writeLines(c(paste0('>',label[m,1]),as.character(label[m,2]),as.character(label[m,3])),sep = "\n",file_name)
    close(file_name)
}
for(m in 1:length(label[,1]))
{
  file_name = file(paste0('C:\\Users/idehz/Desktop/Methylation/Seq/',as.character(label[m,1]),'.name.seq'))
  writeLines(c(paste0('>',label[m,1]),as.character(label[m,2])),sep = "\n",file_name)
  close(file_name)
}
