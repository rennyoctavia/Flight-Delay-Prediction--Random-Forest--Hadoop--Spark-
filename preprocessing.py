from mrjob.job import MRJob
import sys
from datetime import datetime

class MRPreprocessing(MRJob):

    def mapper(self,_,line):
        # remove the "" and split based on comma
        line = line.replace('"', '').split(',')
        line = map(str.strip, line)
	
        # if not the heading line and ARR_DELAY is not null
        # Month (idx: 1) categorical
        # Day_of_month (idx:2) numerical
        # day_of_week, (idx:3) categorical
        # Op_unique_carrier,(4) categorical
        # Tail_num (5) categorical
        # Op_carrier_fl_num (6) categorical
        # Origin (7) categorical
        # Dest (8) categorical
        # Crs_dep_time (9) numerical
        # Crs_arr_time (14) numerical
        # Csr_elapsed_time, (20) numerical
        # Distance (23) numerical
        # dep_delay(target) (11) numerical
        # arr_delay(target) (16) numerical

        if len(line) >= 29 and line[0] != 'YEAR' and line[16] != '' and line[11] !='':
            for i in [2,20,23]:
                try:
                    isnumber = float(line[i])
                except ValueError:
                    line[i]='0.00'

            crs_dep_time = int(line[9][0:2])*60 + int(line[9][2:])
            crs_arr_time = int(line[14][0:2])*60 + int(line[14][2:])
            data_row = [line[1],line[2],line[3],line[4],line[5],line[6],line[7],
                line[8],crs_dep_time,crs_arr_time,line[20],line[23],line[11],line[16]]
            yield _, ','.join(map(str, data_row))

if __name__ == '__main__':
    start_time=datetime.now()
    MRPreprocessing.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    sys.stderr.write(str(elapsed_time))
