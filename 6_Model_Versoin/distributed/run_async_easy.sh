#!/bin/bash
mkdir distributed0
mkdir distributed1
mkdir distributed2
cp asynchronous_update_easy.py distributed0/asynchronous_update_easy.py
cp asynchronous_update_easy.py distributed1/asynchronous_update_easy.py
cp asynchronous_update_easy.py distributed2/asynchronous_update_easy.py
docker-compose up -d
docker-compose exec -d tf0 python asynchronous_update_easy.py --job_name='ps' --task_id=0 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'
docker-compose exec -d tf1 python asynchronous_update_easy.py --job_name='worker' --task_id=0 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'
docker-compose exec -d tf2 python asynchronous_update_easy.py --job_name='worker' --task_id=1 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'