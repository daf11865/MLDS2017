#!/bin/bash

if [ ! -f "datum.tar.gz" ]
then
wget https://gitlab.com/dfke/MLDS2017_datum/raw/master/hw2/datum.tar.gz
fi

if [ ! -d "temp" ]
then
tar -zxvf datum.tar.gz
fi

python main.py --testidlistpath $1 --featpath $2
