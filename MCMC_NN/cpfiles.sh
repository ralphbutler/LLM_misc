
while IFS= read -r filename; do
  cp "uf20-91/$filename" "testdata/"
done < temp2
