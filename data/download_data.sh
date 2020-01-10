# warning: if you cancel this while a file is downloading,
# that file will be corrupt and this script won't try to redownload

for cancer in GBM Breast Colon Kidney Lung; do
  if ! [ -f $cancer.zip ]; then
    echo "downloading $cancer"
    curl -O http://compbio.cs.toronto.edu/SNF/SNF/Software_files/$cancer.zip
  fi
  unzip $cancer.zip -d $cancer
done
