for i in `ls /export/home/s2784083/seq/*.fasta`; do /export/home/s2784083/ncbi-blast-2.2.25+-x64-linux/ncbi-blast-2.2.25+/bin/psiblast -query /export/home/s2784083/seq/$i -db /export/home/s2784083/db1/nr -out $i.out -num_iterations 3 -out_ascii_pssm $i.pssm -inclusion_ethresh 0.001; done




-bash-4.1$ for i in `ls | grep .seq$`; do /export/home/s2878100/source/prolib/ex/contact /export/home/s2784083/Sequences/$i -cutoff 1 `echo /export/home/s2784083/pdb_files/pdb$i.ent | sed 's/..seq//'` > /export/home/s2784083/Sequences/$i.cn ; done

ls | grep .seq$ | more


-bash-4.1$ for i in `ls | grep .seq`; do echo /export/home/s2784083/Sequences/$i `/export/home/s2784083/pdb_files/$i | sed 's/..seq//'`; done 

-bash-4.1$ for i in `ls | grep .seq`; do echo test$i | sed 's/..seq//'; done 

-bash-4.1$ for i in `ls | .seq`; do echo $i; done

-bash-4.1$ /export/home/s2878100/source/prolib/ex/contact -seqcut /export/home/s2784083/Sequences/1a1xA.seq -cutoff 8 -seq0 /export/home/s2784083/pdb_files/pdb1a1x.ent

-bash-4.1$ rm -r *.cn.cn


/home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/E7EUT5 -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db -out E7EUT5.out -num_iterations 3 -out_ascii_pssm E7EUT5.pssm -threshold 0.001



/home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/E7EUT5 -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out E7EUT5.out -num_iterations 3 -out_ascii_pssm E7EUT5.pssm -threshold 0.001


rm -r /home/adehzangi/.synapseCache/*


/home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/Plz1.txt -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out Plz1.out -num_iterations 3 -out_ascii_pssm Plz1.pssm -threshold 0.001 -num_threads 6


/home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.2.28+/bin/psiblast -query /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/Plz1.txt -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out Plz1.out -num_iterations 3 -out_ascii_pssm Plz1.pssm -inclusion_ethresh 0.001



/home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/Plz.txt -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out Plz4.out -num_iterations 3 -out_ascii_pssm Plz4.pssm -inclusion_ethresh 0.001 -num_threads 8


for i in *.seq; do /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Downloads/Protein_database/Butyrylation/Butyrylation/$i -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out $i.out -num_iterations 3 -out_ascii_pssm $i.pssm -inclusion_ethresh 0.001 -num_threads 6; done



for i in *.seq; do /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/bin/psiblast -query /home/adehzangi/Downloads/Protein_database/Butyrylation/Butyrylation/$i -db /home/adehzangi/Desktop/Alignment_Toolbox/PSIBLAST/ncbi-blast-2.3.0+/db/nr -out $i.out -num_iterations 3 -out_ascii_pssm $i.pssm -inclusion_ethresh 0.001 -num_threads 6; done


Try to run HSE pred:

for i in ls *.seq; do /home/adehzangi/Desktop/PHD_Exp/HSEserver/HSEb-pred/v2/ourhsebpred.py full_nn_pssm $i.pssm $i.spd3; done


