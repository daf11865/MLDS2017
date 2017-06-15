#!/bin/bash

if [ ! -f "datum.tar.gz" ]
then
wget https://gitlab.com/dfke/MLDS2017_datum/raw/master/hw4/datum.tar.gz
fi

if [ ! -d "temp" ]
then
tar -zxvf datum.tar.gz
fi

python main.py --model $1 --input $2 --output $3
