FILE=$1

if [[ $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" ]]; then
    echo "Downloadable datasets: monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo"
    exit 1
fi

mkdir datasets
echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE