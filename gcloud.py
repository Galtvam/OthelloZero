#!/usr/bin/env python

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of using the Compute Engine API to create and delete instances.
Creates a new compute engine instance and uses it to apply a caption to
an image.
    https://cloud.google.com/compute/docs/tutorials/python-guide
For more information, see the README.md under /compute.
"""

import os
import time
import tarfile
import logging
import pathlib
import argparse
import subprocess

import paramiko
import googleapiclient.discovery

from paramiko.client import SSHClient



SSH_USER = 'othello-zero'
REMOTE_HOME = f'/home/{SSH_USER}'
DEFAULT_ZONE = 'us-west1-b'
INSTANCE_LABEL = 'othello-zero', 'true'
INSTANCE_NAME = 'othello-zero-worker-{}'
STARTUP_SCRIPT = './gcloud-startup-script.sh'

# [START list_instances]
def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result['items'] if 'items' in result else None
# [END list_instances]


# [START create_instance]
def create_instance(compute, project, zone, name):
    # Get the latest Debian Jessie image.
    image_response = compute.images().getFromFamily(
        project='ubuntu-os-cloud', family='ubuntu-2004-lts').execute()
    source_disk_image = image_response['selfLink']

    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
    startup_script = open(
        os.path.join(
            os.path.dirname(__file__), STARTUP_SCRIPT), 'r').read()

    config = {
        'name': name,
        'machineType': machine_type,

        # Specify the boot disk and the image to use as a source.
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': source_disk_image,
                }
            }
        ],

        'labels': {
            INSTANCE_LABEL[0]: INSTANCE_LABEL[1]
        },

        # Specify a network interface with NAT to access the public
        # internet.
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],

        # Allow the instance to access cloud storage and logging.
        'serviceAccounts': [{
            'email': 'default',
            'scopes': [
                'https://www.googleapis.com/auth/devstorage.read_write',
                'https://www.googleapis.com/auth/logging.write'
            ]
        }],

        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        'metadata': {
            'items': [{
                # Startup script is automatically executed by the
                # instance upon startup.
                'key': 'startup-script',
                'value': startup_script
            }]
        }
    }

    return compute.instances().insert(
        project=project,
        zone=zone,
        body=config).execute()
# [END create_instance]


# [START delete_instance]
def delete_instance(compute, project, zone, name):
    return compute.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()
# [END delete_instance]


# [START search_instance]
def search_instances(compute, project, zone, label_key, label_value):
    result = compute.instances().list(
        project=project,
        zone=zone,
        filter=f'labels.{label_key}={label_value}').execute()
    return result['items'] if 'items' in result else []
# [END delete_instance]


# [START get_instance]
def get_instance(compute, project, zone, instance_name):
    result = compute.instances().get(
        project=project,
        zone=zone,
        instance=instance_name).execute()
    return result
# [END get_instance]


# [START stop_instance]
def stop_instance(compute, project, zone, instance_name):
    result = compute.instances().stop(
        project=project,
        zone=zone,
        instance=instance_name).execute()
    return result
# [END stop_instance]


# [START restart_instance]
def restart_instance(compute, project, zone, instance_name):
    result = compute.instances().reset(
        project=project,
        zone=zone,
        instance=instance_name).execute()
    return result
# [END restart_instance]


# [START start_instance]
def start_instance(compute, project, zone, instance_name):
    result = compute.instances().start(
        project=project,
        zone=zone,
        instance=instance_name).execute()
    return result
# [END start_instance]


# [START wait_for_operation]
def wait_for_operation(compute, project, zone, operation):
    print('Waiting for operation to finish...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)
# [END wait_for_operation]


def self_upload_to_instance(instance, key_filename):
    print('uploading projects files to instance...')

    ip = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    
    files = subprocess.check_output('git ls-files', shell=True, text=True).splitlines()
    
    client = ssh_connection(ip, key_filename)
    sftp = client.open_sftp()
    remote_filepath = os.path.join(REMOTE_HOME, 'files.tar.xz')
    tar_fileobj = sftp.open(remote_filepath, mode='w')
    tar_file = tarfile.open(fileobj=tar_fileobj, mode='w:xz')
    for filepath in files:
        local_path = os.path.join(os.getcwd(), filepath)
        tar_file.add(local_path, arcname=filepath)
    tar_file.close()
    tar_fileobj.close()
    sftp.close()
    client.exec_command(f'tar -xf files.tar.xz')
    client.exec_command(f'rm files.tar.xz')
    client.close()

def get_instance_external_ip(instance):
    return instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']


def get_instance_internal_ip(instance):
    return instance['networkInterfaces'][0]['networkIP']


def wait_for_instance_startup_script(instance, key_filename):
    ip = get_instance_external_ip(instance)
    client = ssh_connection(ip, key_filename)
    stdin, stdout, stderr = client.exec_command(
        'ps aux | grep -v grep | grep "/bin/bash /startup" | awk \'{print $2}\'')
    lines = stdout.readlines()
    if not lines:
        return
    pid = lines[0].strip()
    stdin, stdout, stderr = client.exec_command(f'sudo cat /proc/{pid}/fd/1 > /dev/null')
    stdout.channel.recv_exit_status()


def ssh_connection(ip, key_filename, timeout=120, auth_timeout=60):
    client = SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=SSH_USER, key_filename=key_filename, timeout=timeout, auth_timeout=auth_timeout)
    return client


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('project', help='Google Cloud Platform project name')
    parser.add_argument('credentials', help='Google Cloud API Credentials JSON file path')
    parser.add_argument('-z', '--zone', default=DEFAULT_ZONE, help='Google Cloud Platform instances zone')
    parser.add_argument('-k', '--key-filename', default=None, help='SSH Auth key file')
    
    subprasers = parser.add_subparsers(dest='command')
    
    create = subprasers.add_parser('create', help='Create new worker instances')
    create.add_argument('instances_amount', nargs='?', default=1, type=int, help='Number of instances to create')
    
    delete = subprasers.add_parser('delete', help='Delete worker instances')
    delete.add_argument('instances_amount', nargs='?', default=None, type=int, 
                        help='Number of instances to delete. If None, delete all instances.')

    list_ = subprasers.add_parser('list', help='List worker instances')

    upload_env = subprasers.add_parser('upload', help='Uploads environment files (project files)')

    start = subprasers.add_parser('start', help='Start all Google Cloud workers')

    stop = subprasers.add_parser('stop', help='Stop all Google Cloud workers')

    restart = subprasers.add_parser('restart', help='Restart all Google Cloud workers')

    args = parser.parse_args()

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials

    compute = googleapiclient.discovery.build('compute', 'v1')

    instances = search_instances(compute, args.project, args.zone, *INSTANCE_LABEL)

    if args.command == 'list':
        print(' '.join(instance['name'] for instance in instances))
    elif args.command == 'create':
        operations = []
        for i in range(len(instances), len(instances) + args.instances_amount):
            operation = create_instance(compute, args.project, args.zone, INSTANCE_NAME.format(i))
            operations.append(operation)
        for operation in operations:
            wait_for_operation(compute, args.project, args.zone, operation['name'])

        assert args.key_filename, 'Cannot upload environment files without key file'

        print('waiting one minute to instances be accessible...')
        time.sleep(60)

        old_instances = [instance['name'] for instance in instances]
        new_instances = search_instances(compute, args.project, args.zone, *INSTANCE_LABEL)
        new_instances = [instance for instance in new_instances if instance['name'] not in old_instances]
        
        for instance in new_instances:
            self_upload_to_instance(instance, args.key_filename)
        
        for instance in new_instances:
            print('waiting for startup script finish to run')
            wait_for_instance_startup_script(instance, args.key_filename)

    elif args.command == 'delete':
        instances.reverse()
        amount = args.instances_amount or len(instances)
        operations = []
        for instance, _ in zip(instances, range(amount)):
            operation = delete_instance(compute, args.project, args.zone, instance['name'])
            operations.append(operation)
        
        for operation in operations:
            wait_for_operation(compute, args.project, args.zone, operation['name'])

    elif args.command == 'upload':
        assert args.key_filename, 'Cannot upload environment files without key file'

        for instance in instances:
           self_upload_to_instance(instance, args.key_filename)

    elif args.command == 'start':
        for instance in instances:
           start_instance(compute, args.project, args.zone, instance['name'])
    
    elif args.command == 'stop':
        for instance in instances:
           stop_instance(compute, args.project, args.zone, instance['name'])
    
    elif args.command == 'restart':
        for instance in instances:
           restart_instance(compute, args.project, args.zone, instance['name'])
