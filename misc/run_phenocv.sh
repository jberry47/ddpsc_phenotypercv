#!/bin/bash
printf "\nProcessing VIS Images...\n"
find Images/ -name "VIS_SV*" | xargs -I{} -P10 /home/jberry/programs/PhenotyperCV/PhenotyperCV -m=VIS_CH -i={} -b=average_images_pheno5_vis.png -s=pheno5_shapes.txt -c=pheno5_color.txt -d=pheno5_leaves.txt

printf "\nProcessing NIR Images...\n"
find Images/ -name "NIR_SV*" | xargs -I{} -P10 /home/jberry/programs/PhenotyperCV/PhenotyperCV -m=NIR -i={} -b=average_images_nir.png -c=nir_color.txt
