# SETTING OPTION FLAGS
while getopts i:p: option
do
case "${option}"
in
i) INDEX=${OPTARG};;
p) PIZERO=${OPTARG};;
esac
done

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Give -p for RPi and -i for img index"
    echo "-p 0  # for RPi3."
    echo "-p 1  # for top Pi0."
    echo "-p 2  # for bottom Pi0"
    exit 1
fi

#PIZERO=$1
#INDEX=$2

if [ $PIZERO = 0 ]; then
    # name of calib picture
    echo 'Connecting to PiZERO and iterating for finding white balance'
    python3 white_balance_iteration.py 
    echo 'Done.'
    exit 0
fi

# name of calib picture
PICTURE_NAME='pizero_'$PIZERO'_whitebalance_'$INDEX'.jpg'

echo 'Connecting to PiZERO and iterating for finding white balance'
ssh -tTo ServerAliveInterval=60 pi@10.0.1$PIZERO.2 'python3 white_balance_iteration.py' 

#echo 'Transferring picture from PiZERO'
#rsync -a pi@10.0.1$PIZERO.2:/home/pi/$PICTURE_NAME calib_images/ &
echo 'Done.'

exit 0

