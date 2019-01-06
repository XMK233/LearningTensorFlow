#!/bin/bash
mkdir distributed0
mkdir distributed1
mkdir distributed2
cp async.py distributed0/async.py
cp async.py distributed1/async.py
cp async.py distributed2/async.py
docker-compose up -d
docker-compose exec -d tf0 python async.py --job_name='ps' --task_id=0 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'
docker-compose exec -d tf1 python async.py --job_name='worker' --task_id=0 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'
docker-compose exec -d tf2 python async.py --job_name='worker' --task_id=1 --ps_hosts='tf0:2222' --worker_hosts='tf1:2222,tf2:2222'