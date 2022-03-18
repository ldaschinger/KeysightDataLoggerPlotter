"""
Extracts various values from Keysight Data Logger *.csv file like
average current, average power, cumulative sum of energy used

Possibility to plot the current and energy consumption with matplotlib
"""

__author__ = "Lukas Daschinger"
__version__ = "1.0.1"
__maintainer__ = "Lukas Daschinger"
__email__ = "ldaschinger@student.ethz.ch"


import getopt
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re


def analyzeLoggerData(filepath):
    # print head including sampling interval
    with open(filepath) as myfile:
        head = [next(myfile) for x in range(6)]
    # print(head, "\n")

    #time tick spacing in seconds
    xtick_spacing_s = 20
    ytick_spacing_s = 50

    ymin = 0
    ymax = 900

    # define subsection in seconds for calculation only
    startSeconds = 0
    endSeconds = 50

    # # define subsection in seconds for calculation and graph
    # startSecondsSnipping = 5
    # endSecondsSnipping = 50

    linewidthHlines = 0.4
    linewidthGraph = 0.2

    hlines = [0, 50, 100, 150, 200]
    # hlines = [5, 10, 15, 20, 25]

    # detect the used sampling interval
    period_in_s = float(re.findall("\d+\.\d+", head[2])[0])

    # df is a pandas series
    # df = pd.read_csv(r'/Users/asdf/Desktop/TEST/dlog1.csv', sep=",", skiprows=6)
    df = pd.read_csv(filepath, sep=",", skiprows=6)
    # print(df)

    index = df.index
    number_of_rows = len(index)
    # print number of samples and duration of measurement
    # print("number of rows: ", number_of_rows)
    # print("measurement duration [s]: ", number_of_rows*period_in_s, "\n")

    # # use a subsection of the trace
    # startSampleSnipping = int(startSecondsSnipping/period_in_s)
    # endSampleSnipping = int(endSecondsSnipping/period_in_s)
    # df['Curr avg 1'] = df['Curr avg 1'][startSampleSnipping:endSampleSnipping]


    # print some stats to understand if values are valid
    meanV = df['Volt avg 1'].mean()
    maxV = df['Volt avg 1'].max()
    minV = df['Volt avg 1'].min()
    standardDevV = df['Volt avg 1'].std()
    meanC = df['Curr avg 1'].mean()
    standardDevI = df['Curr avg 1'].std()

    """
    Given that the voltage is relatively stable (normal stddev around 6mV) 
    we can look at the current stddev directly to get a good estimate of the power standard deviation
    If the voltage changes strongly together with the current we also have to consider power stddev
    """


    startSample = int(startSeconds/period_in_s)
    endSample = int(endSeconds/period_in_s)

    meanCSubsection = df['Curr avg 1'][startSample:endSample].mean()

    # print(str(meanC*1000))
    # print(str(standardDevI*1000) + "\n")

    # we calculate the power consumed at every sampling interval by using I*A
    df['Power'] = df['Curr avg 1'] * df['Volt avg 1']
    meanP = df['Power'].mean()
    standardDevP = df['Power'].std()

    # to get the total Energy used for the specific experiment use E = P*t
    df['Energy'] = df['Power']*period_in_s
    totalE = df['Energy'].sum()

    # numeric integration over the Power values
    # to calculate integral (Energy used) between two samples a and b do:  df['Energy Sum'][b] - df['Energy Sum'][a]
    df['Energy Sum'] = df['Energy'].cumsum()
    # print('total Energy by cumsum [J]: ' + str(df['Energy Sum'].iloc[-1]) + '\n')


    # https://stackoverflow.com/questions/66273368/how-to-plot-a-single-column-graph-of-a-csv-file-in-python
    # timestampsCurrent = [i * period_in_s for i in range(int(len(df['Curr avg 1'])/1000)+1)]
    timestampsCurrent = [i * period_in_s for i in range(len(df['Curr avg 1']))]
    plt.xlabel('time [s]',fontsize=14)
    plt.ylabel('Current [mA]',fontsize=14)
    # plt.title('Current')

    mACurrent = (df['Curr avg 1'])*1000
    # mACurrentDownsampled = (df['Curr avg 1'][np.arange(len(df['Curr avg 1'])) % 1000 == 1])*1000

    plt.xticks(np.arange(min(timestampsCurrent), max(timestampsCurrent) + 1, xtick_spacing_s))
    plt.yticks(np.arange(0, max(mACurrent) + 1, ytick_spacing_s))

    plt.ylim(ymin, ymax)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.plot(timestampsCurrent, mACurrent,  color='#33ccff', linewidth=linewidthGraph)
    plt.hlines(hlines, min(timestampsCurrent), max(timestampsCurrent), color='grey', linewidth=linewidthHlines)
    # plt.show()


    # fig, axs = plt.subplots(2)
    # fig.suptitle('Current and accumulated Energy')
    # timestampsCurrent = [i * period_in_s for i in range(len(df['Curr avg 1']))]
    # axs[0].plot(timestampsCurrent, df['Curr avg 1'])
    # axs[0].set_title('Current')
    # axs[0].set(xlabel = 'time [s]', ylabel = 'Current [mA]')
    #
    # timestampsEnergy = [i * period_in_s for i in range(len(df['Energy Sum']))]
    # axs[1].plot(timestampsEnergy, df['Energy Sum'])
    # axs[1].set_title('Energy used')
    # axs[1].set(xlabel = 'time [s]', ylabel = 'total Energy [mJ]')
    # fig.tight_layout()
    # plt.show()

    return meanC*1000


def analyzeTestCustom(folderpath, bitrate, res1, fps1, codec1, res2="null", fps2="null", codec2="null",
                      res3="null", fps3="null", codec3="null", res4="null", fps4="null", codec4="null",
                      res5="null", fps5="null", codec5="null"):

    settings = []

    dictlist = [dict() for x in range(5)]
    dictlist[0] = {"res": res1, "fps": fps1, "codec": codec1}
    dictlist[1] = {"res": res2, "fps": fps2, "codec": codec2}
    dictlist[2] = {"res": res3, "fps": fps3, "codec": codec3}
    dictlist[3] = {"res": res4, "fps": fps4, "codec": codec4}
    dictlist[4] = {"res": res5, "fps": fps5, "codec": codec5}

    folderpaths = []
    for i in range(5):
        folderpaths.append(folderpath + dictlist[i].get("codec") + "/" + bitrate + "/" + bitrate + dictlist[i].get("res") + dictlist[i].get("fps"))

    csvMeans = [[0 for x in range(0)] for y in range(5)]
    for i in range(5):
        # if we have varying number of tests and therefore .csv files available we must find all in the folder
        if dictlist[i].get("res") != "null":
            for filename in os.listdir(folderpaths[i]):
                name, extension = os.path.splitext(filename)
                if extension == ".csv":
                    csvMeans[i].append(analyzeLoggerData(folderpaths[i] + "/" + filename))

    # # if we have varying number of tests and therefore .csv files available we must find all in the folder
    # for filename in os.listdir(folderpath1):
    #     name, extension = os.path.splitext(filename)
    #     if extension == ".csv":
    #         mean1.append(analyzeLoggerData(folderpath1 + "/" + filename))
    # npMean1 = np.asarray(mean1)

    npMeans = [np.empty([5]) for x in range(5)]
    for i in range(5):
        # if we have varying number of tests and therefore .csv files available we must find all in the folder
        if dictlist[i].get("res") != "null":
            npMeans[i] = np.asarray(csvMeans[i])


    for i in range(5):
        # if we have varying number of tests and therefore .csv files available we must find all in the folder
        if dictlist[i].get("res") != "null":
            print(str(format(npMeans[i].mean(), ".2f")) + " " + str(format(npMeans[i].std(), ".2f")) + " " + str(format(npMeans[i].std(), ".2f")) + "  ", end="", flush=True)

    print("\n")
    # print(str(format(npMean1.mean(), ".2f")) + " " + str(format(npMean1.std(), ".2f")) + " " + str(format(npMean1.std(), ".2f")) + "  " +
    #       str(format(npMean2.mean(), ".2f")) + " " + str(format(npMean2.std(), ".2f")) + " " + str(format(npMean2.std(), ".2f")) + "  " +
    #       str(format(npMean3.mean(), ".2f")) + " " + str(format(npMean3.std(), ".2f")) + " " + str(format(npMean3.std(), ".2f")) + "  " +
    #       str(format(npMean4.mean(), ".2f")) + " " + str(format(npMean4.std(), ".2f")) + " " + str(format(npMean4.std(), ".2f")) + "  " +
    #       str(format(npMean5.mean(), ".2f")) + " " + str(format(npMean5.std(), ".2f")) + " " + str(format(npMean5.std(), ".2f")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folderpath",
                        required=True,
                        default=None,
                        help="Path to target CSV file folder")

    args = parser.parse_args()

    """
    directory structure:
    folderpath
        folderpath_small/large/auto
            dlog1.csv/dlog2.csv/dlog3.csv/dlog4.csv
    """

    # # 30fps tests
    # analyzeTest(args.folderpath, bitrate="300", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="600", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="900", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="1300", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="1800", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="2700", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="4000", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="4750", fps="30", codec="H264")
    # analyzeTest(args.folderpath, bitrate="6000", fps="30", codec="H264")

    # # 15fps tests
    # analyzeTest(args.folderpath, bitrate="600", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="900", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="1300", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="1800", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="2700", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="4000", fps="15", codec="H264")
    # analyzeTest(args.folderpath, bitrate="6000", fps="15", codec="H264")

    # VP8 tests
    # analyzeTest(args.folderpath, bitrate="900", fps="30", codec="VP8")
    # analyzeTest(args.folderpath, bitrate="1800", fps="30", codec="VP8")
    # analyzeTest(args.folderpath, bitrate="4000", fps="30", codec="VP8")
    # analyzeTest(args.folderpath, bitrate="6000", fps="30", codec="VP8")

    # # VP8 tests
    # analyzeTestCustom(args.folderpath, bitrate="900",
    #                   res1="_small_", fps1="30", codec1="VP8",
    #                   res2="_large_", fps2="30", codec2="VP8",
    #                   res3="_auto_", fps3="30", codec3="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="1800",
    #                   res1="_small_", fps1="30", codec1="VP8",
    #                   res2="_large_", fps2="30", codec2="VP8",
    #                   res3="_auto_", fps3="30", codec3="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="4000",
    #                   res1="_small_", fps1="30", codec1="VP8",
    #                   res2="_large_", fps2="30", codec2="VP8",
    #                   res3="_auto_", fps3="30", codec3="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="6000",
    #                   res1="_small_", fps1="30", codec1="VP8",
    #                   res2="_large_", fps2="30", codec2="VP8",
    #                   res3="_auto_", fps3="30", codec3="VP8")

    # VP8 vs H264 tests
    # analyzeTestCustom(args.folderpath, bitrate="900",
    #                   res1="_small_", fps1="30", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="VP8",
    #                   res3="_large_", fps3="30", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="1800",
    #                   res1="_small_", fps1="30", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="VP8",
    #                   res3="_large_", fps3="30", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="4000",
    #                   res1="_small_", fps1="30", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="VP8",
    #                   res3="_large_", fps3="30", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="VP8")
    # analyzeTestCustom(args.folderpath, bitrate="6000",
    #                   res1="_small_", fps1="30", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="VP8",
    #                   res3="_large_", fps3="30", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="VP8")

    # 15fps vs 30fps H264
    # analyzeTestCustom(args.folderpath, bitrate="600", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="900", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="1300", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="1800", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="2700", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="4000", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")
    # analyzeTestCustom(args.folderpath, bitrate="6000", res1="_small_", fps1="15", codec1="H264",
    #                   res2="_small_", fps2="30", codec2="H264", res3="_large_", fps3="15", codec3="H264",
    #                   res4="_large_", fps4="30", codec4="H264")

    # PC only test
    analyzeTestCustom(args.folderpath, bitrate="nolimit",
                      res1="_480_", fps1="10", codec1="VP8",
                      res2="_480_", fps2="30", codec2="VP8",)
    analyzeTestCustom(args.folderpath, bitrate="nolimit",
                      res1="_720_", fps1="10", codec1="VP8",
                      res2="_720_", fps2="30", codec2="VP8", )
    analyzeTestCustom(args.folderpath, bitrate="nolimit",
                      res1="_1080_", fps1="10", codec1="VP8",
                      res2="_1080_", fps2="30", codec2="VP8", )

    analyzeTestCustom(args.folderpath, bitrate="nolimit",
                      res1="_720_", fps1="10", codec1="H264",
                      res2="_720_", fps2="30", codec2="H264", )