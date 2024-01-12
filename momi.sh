#!/bin/bash

# Feature Extraction
cd Kowalski

read -p "Enter the project folder path: " projectFolderPath
echo "$projectFolderPath" > /tmp/projectFolderPath.txt
# projectFolderPath="/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Population/spring-petclinic-microservices"
modifiedFolderPath=$(echo "$projectFolderPath" | sed 's/\/Population\//\/Decomposition\/inputs\//')
jsonFilePath="$modifiedFolderPath.json"

mvn clean install && java -jar ./target/Kowalski-1.0-SNAPSHOT-jar-with-dependencies.jar 

# Decomposition
cd ../Decomposition
python3 momi.py "$jsonFilePath"

echo "Decomposition Complete"