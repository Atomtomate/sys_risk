set terminal svg enhanced background rgb 'white' size 720,720
set output "dist.svg"
plot    "< awk '{if($3 == \"0\") print}' dist.txt" u 1:2 t "dd" w p pt 7 ps 0.4, \
        "< awk '{if($3 == \"1\") print}' dist.txt" u 1:2 t "ds" w p pt 7 ps 0.4, \
        "< awk '{if($3 == \"2\") print}' dist.txt" u 1:2 t "sd" w p pt 7 ps 0.4, \
        "< awk '{if($3 == \"3\") print}' dist.txt" u 1:2 t "ss" w p pt 7 ps 0.4, \
        "< awk '{if($3 == \"4\") print}' dist.txt" u 1:2 t "ERROR" w p pt 7 ps 0.4
