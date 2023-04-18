#!/bin/bash
# Get all port numbers in use by Streamlit and output the URLs for all running streamlit apps. 

CURRENTDATE=`date +"%Y-%m-%d %T"`
RED='\033[0;31m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
NC='\033[0m'
S3_PATH=$1


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


# Find the process IDs of all running Streamlit instances
streamlit_pids=$(pgrep streamlit)

# Initialize an empty array to hold the port numbers
port_array=()
# Loop through each process ID and find the port number in use
for pid in $streamlit_pids; do
    # Get the port number for the process
    port=$(lsof -a -i -P -n -p $pid | grep LISTEN | awk -F':' '{print $2}' | cut -d ' ' -f 1)\
    # port=$(lsof -a -i -P -n -p $pid | grep ".*:\d\+.*(LISTEN)$" | awk -F':' '{print $2}' | cut -d ' ' -f 1)
    port="${port:0:4}"
    
    # Add the port number to the array
    port_array+=("$port")
    
done


echo "These are the Streamlit Apps Currently Running: "
# Loop through the array and print each port number
for PORT in "${port_array[@]}"; do
    
    # if it's not a collaborative space 
    if [ -z "$SPACE_NAME" ] || [ $SPACE_NAME == "null" ] ;
    then
        # If it's a user-profile access
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

    link="${STUDIO_URL}/jupyter/${RESOURCE_NAME}/proxy/${PORT}/"

    echo -e "${GREEN}${link}${NC}"
done
exit 0
fi




