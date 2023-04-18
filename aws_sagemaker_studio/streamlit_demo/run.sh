#!/bin/sh
CURRENTDATE=`date +"%Y-%m-%d %T"`
RED='\033[0;31m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
NC='\033[0m'
S3_PATH=$1

# Run the Streamlit app and save the output to "temp.txt"
streamlit run app.py > temp.txt & 

# Read the text file using cat
echo "Getting the URL to view your Streamlit app in the browser"

# Extract the last four digits of the port number from the Network URL
sleep 5
PORT=$(grep "Network URL" temp.txt | awk -F':' '{print $NF}' | awk '{print $1}' | tail -c 5)
echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Port Number ${PORT}" 



# Get Studio domain information
DOMAIN_ID=$(jq .DomainId /opt/ml/metadata/resource-metadata.json || exit 1)
RESOURCE_NAME=$(jq .ResourceName /opt/ml/metadata/resource-metadata.json || exit 1)
RESOURCE_ARN=$(jq .ResourceArn /opt/ml/metadata/resource-metadata.json || exit 1)

# Remove quotes from string
DOMAIN_ID=`sed -e 's/^"//' -e 's/"$//' <<< "$DOMAIN_ID"`
RESOURCE_NAME=`sed -e 's/^"//' -e 's/"$//' <<< "$RESOURCE_NAME"`
RESOURCE_ARN=`sed -e 's/^"//' -e 's/"$//' <<< "$RESOURCE_ARN"`
RESOURCE_ARN_ARRAY=($(echo "$RESOURCE_ARN" | tr ':' '\n'))

# Get Studio domain region
REGION=$(echo "${RESOURCE_ARN_ARRAY[3]}")

# Check if it's Collaborative Space
SPACE_NAME=$(jq .SpaceName /opt/ml/metadata/resource-metadata.json || exit 1)

# if it's not a collaborative space 
if [ -z "$SPACE_NAME" ] || [ $SPACE_NAME == "null" ] ;
then
    # If it's a user-profile access
    echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Domain Id ${DOMAIN_ID}"
    STUDIO_URL="https://${DOMAIN_ID}.studio.${REGION}.sagemaker.aws"
    
# It is a collaborative space
else

    SEM=true
    SPACE_ID=

    # Check if Space Id was previously configured
    if [ -f /tmp/space-metadata.json ]; then
        SAVED_SPACE_ID=$(jq .SpaceId /tmp/space-metadata.json || exit 1)
        SAVED_SPACE_ID=`sed -e 's/^"//' -e 's/"$//' <<< "$SAVED_SPACE_ID"`

        if [ -z "$SAVED_SPACE_ID" ] || [ $SAVED_SPACE_ID == "null" ]; then
            ASK_INPUT=true
        else
            ASK_INPUT=false
        fi
    else
        ASK_INPUT=true
    fi

    # If Space Id is not available, ask for it
    while [[ $SPACE_ID = "" ]] ; do
        # If Space Id already configured, skeep the ask
        if [ "$ASK_INPUT" = true ]; then
            echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Please insert the Space Id from your url. e.g. https://${GREEN}<SPACE_ID>${NC}.studio.${REGION}.sagemaker.aws/jupyter/default/lab"
            read SPACE_ID
            SEM=true
        else
            SPACE_ID=$SAVED_SPACE_ID
        fi

        if ! [ -z "$SPACE_ID" ] && ! [ $SPACE_ID == "null" ] ;
        then
            while $SEM; do
                echo "${SPACE_ID}"
                read -p "Should this be used as Space Id? (y/N) " yn
                case $yn in
                    [Yy]* )
                        echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Domain Id ${DOMAIN_ID}"
                        echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Space Id ${SPACE_ID}"

                        jq -n --arg space_id $SPACE_ID '{"SpaceId":$space_id}' > /tmp/space-metadata.json

                        STUDIO_URL="https://${SPACE_ID}.studio.${REGION}.sagemaker.aws"

                        SEM=false
                        ;;
                    [Nn]* ) 
                        SPACE_ID=
                        ASK_INPUT=true
                        SEM=false
                        ;;
                    * ) echo "Please answer yes or no.";;
                esac
            done
        fi
    done
fi

echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Studio Url ${STUDIO_URL}"


link="${STUDIO_URL}/jupyter/${RESOURCE_NAME}/proxy/${PORT}/"

echo -e "${CYAN}${CURRENTDATE}: [INFO]:${NC} Starting Streamlit App"
echo -e "${CYAN}${CURRENTDATE}: [INFO]: ${GREEN}${link}${NC}"

exit 0
fi
