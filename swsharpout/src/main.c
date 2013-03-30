#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/swsharp.h"

static struct option options[] = {
    {"input", required_argument, 0, 'i'},
    {"plot", required_argument, 0, 'p'},
    {"out", required_argument, 0, 'o'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static void help();

int main(int argc, char* argv[]) {

    char* alignmentPath = NULL;
    char* plot = NULL;
    char* out = NULL;
    
    while (1) {

        char argument = getopt_long(argc, argv, "i:h", options, NULL);

        if (argument == -1) {
            break;
        }

        switch (argument) {
        case 'i':
            alignmentPath = optarg;
            break;
        case 'p':
            plot = optarg;
            break;
        case 'o':
            out = optarg;
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
    
    if (out != NULL) {
        outputPair(alignment, out); 
    }
    
    if (plot != NULL) {
        outputPlot(alignment, plot);   
    }
    
    chainDelete(alignmentGetQuery(alignment));
    chainDelete(alignmentGetTarget(alignment));
    scorerDelete(alignmentGetScorer(alignment));

    alignmentDelete(alignment);

    return 0;
}

static void help() {
    printf("help\n");
}
