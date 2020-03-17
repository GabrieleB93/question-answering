import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tqdm

tmp_path = "tmp.jsonl"


def dict_size(d):
    """
    This function returns the size in byte of a dictionary.
    Notice that it needs to dump it first in order to get the real
    size of the object when saved onto file

    @param d the dictionary object

    @return the size in Byte
    """
    with open(tmp_path, 'w+') as tmp:
        json.dump(d, tmp)
    size = os.path.getsize(tmp_path)
    #print(size)
    os.remove(tmp_path)
    return size


def split_json(path, dest_folder, batch_size, max_size):
    """
    This function splits jsonline file into chunks that contain
    batch_size lines each and do not exceed max_size.
    In the case in which the batch size leads to a too big json file
    the size constraint is preferred.

    @param path string containing the path to the file
    @param dest_folder path fo the destination folder
    @param batch_size the number of lines that should be contained
        in each small json file
    @param max_size maximum size of each json file
    """

    #If destination folder already exists delete and recreate it
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)
    batch_id = 0
    iter_batch = 1
    num_lines = 0
    with open(path) as infile:
        for line in infile:
            #print(line[0], line[-2])
            line_dict = json.loads(line)
            line_dict_size = dict_size(line_dict)
            if (line_dict_size > max_size):
                print("Error, max size smaller than line size")
                break
            #If destination file is not present create it
            curr_dest = dest_folder + str(batch_id) + ".jsonl"
            if not os.path.exists(curr_dest):
                Path(curr_dest).touch()
                #curr_dest.touch()
            curr_dest_size = os.path.getsize(curr_dest)
            #print("Size of dest file: {}".format(curr_dest_size))
            #If size threshold has been reached before
            #exhausting the batch_size or the batch size was reached
            #set counters to 0 and dump json file
            if (curr_dest_size + line_dict_size >
                    max_size) or (iter_batch == batch_size):
                batch_id += 1
                curr_dest = dest_folder + str(batch_id) + ".jsonl"
                with open(curr_dest, 'a+') as dest:
                    json.dump(line_dict, dest)
                    #Add new line
                    dest.write('\n')
                if (curr_dest_size + line_dict_size > max_size):
                    print("------->File of {} rows created".format(iter_batch))
                else:
                    print("->File of {} rows created".format(iter_batch))
                iter_batch = 1

            else:
                iter_batch += 1
                #Using write mode, since file already exists (l.51)
                with open(curr_dest, 'a') as dest:
                    json.dump(line_dict, dest)
                    dest.write('\n')
            num_lines += 1
        #The last line was written, print #lines of last file
        print("->File of {} rows created".format(iter_batch - 1))

    print("Total number of lines of original file: {}".format(num_lines))


parser = argparse.ArgumentParser(
    description='Tool for splitting large json files.')

parser.add_argument("--input-file",
                    metavar="input_file",
                    default="simplified-nq-train.jsonl",
                    help="Path of input json file")

parser.add_argument("--batch-size",
                    metavar="bs",
                    default=500000,
                    help="Number of lines contained in a minibatch")

parser.add_argument("--file-max-size",
                    metavar="max_size",
                    default=1000,
                    help="Maximum size of file in MB")

parser.add_argument("--dest-folder",
                    metavar="dest_folder",
                    default="small_jsons/",
                    help="Path of destination folder of small jsons")

args = vars(parser.parse_args())

dest_folder = args['dest_folder'].split('/')[0] + "/"
maximum_size = 1000000 * args['file_max_size']

split_json(args['input_file'], dest_folder, args['batch_size'], maximum_size)
