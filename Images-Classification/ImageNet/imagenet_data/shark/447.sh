file="urls.txt"
while IFS= read line
do
        # display $line or do somthing with $line
	echo "$line"
	wget $line --timeout=1 --tries=1
done <"$file"

jpeginfo $(find . -name "*jpg") -d #Remove corrupted images
