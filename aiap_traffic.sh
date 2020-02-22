#!/bin/sh
echo -e "\e[1mGenerating containers to deploy XGboost model\e[0m"
echo " "

docker-compose -f aiap_traffic.yml up -d

while true; do
	echo " "
	echo -e "\e[1mRun the following commands to check the containers:\e[0m"
	echo -e "check status of containers: \e[1mdocker container ls\e[0m"
	echo -e "check model output: \e[1mdocker container logs -f myapp_aiap_mlmodel\e[0m"
	echo -e "clean up all containers: \e[1mdocker-compose -f aiap_traffic.yml down\e[0m"
	echo -e "run your own commands: \e[1m owninput\e[0m"
	echo " "
	read usercommand

	if [ "$usercommand" == "docker container ls" ]; then
		$usercommand
	elif [ "$usercommand" == "docker container logs -f myapp_aiap_mlmodel" ]; then
		$usercommand
	elif [ "$usercommand" == "docker-compose -f aiap_traffic.yml down" ]; then
		$usercommand
		echo -e "\e[1mContainers cleaned up\e[0m"
		break
	elif [ "$usercommand" == "owninput" ]; then
		break
	else 
		echo " "
		echo -e "\e[1mcommand not registered, please re-enter\e[0m"
	fi
done
