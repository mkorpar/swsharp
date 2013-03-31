#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/swsharp.h"

#define OUT_FORMATS_LEN (sizeof(outFormats) / sizeof(CharInt))

typedef struct CharInt {
    const char* format;
    const int code;
} CharInt;

static struct option options[] = {
    {"input", required_argument, 0, 'i'},
    {"out", required_argument, 0, 'o'},
    {"outfmt", required_argument, 0, 't'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static CharInt outFormats[] = {
    { "pair", SW_OUT_PAIR },
    { "pair-stat", SW_OUT_STAT_PAIR },
    { "plot", SW_OUT_PLOT },
    { "stat", SW_OUT_STAT }
};

static int getOutFormat(char* optarg);

static void help();

int main(int argc, char* argv[]) {

    char* alignmentPath = NULL;
    
    char* out = NULL;
    int outFormat = SW_OUT_STAT_PAIR;
    
    while (1) {

        char argument = getopt_long(argc, argv, "i:h", options, NULL);

        if (argument == -1) {
            break;
        }

        switch (argument) {
        case 'i':
            alignmentPath = optarg;
            break;
        case 'o':
            out = optarg;
            break;
        case 't':
            outFormat = getOutFormat(optarg);
            break;
        case 'h':
        default:
            help();
            return -1;
        }
    }
    
    ASSERT_CALL(alignmentPath != NULL, help, "missing option -i (alignment file)");
    
    Alignment* alignment = readAlignment(alignmentPath);
    ASSERT(checkAlignment(alignment), "invalid align");
    
    outputAlignment(alignment, out, outFormat);
    
    chainDelete(alignmentGetQuery(alignment));
    chainDelete(alignmentGetTarget(alignment));
    scorerDelete(alignmentGetScorer(alignment));

    alignmentDelete(alignment);

    return 0;
}

static int getOutFormat(char* optarg) {

    int i;
    for (i = 0; i < OUT_FORMATS_LEN; ++i) {
        if (strcmp(outFormats[i].format, optarg) == 0) {
            return outFormats[i].code;
        }
    }

    ERROR("unknown out format %s", optarg);
}

static void help() {
    printf("help\n");
}
