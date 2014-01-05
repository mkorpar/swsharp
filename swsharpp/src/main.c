#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/swsharp.h"

#define ASSERT(expr, fmt, ...)\
    do {\
        if (!(expr)) {\
            fprintf(stderr, "[ERROR]: " fmt "\n", ##__VA_ARGS__);\
            exit(-1);\
        }\
    } while(0)

#define CHAR_INT_LEN(x) (sizeof(x) / sizeof(CharInt))

typedef struct CharInt {
    const char* format;
    const int code;
} CharInt;

static struct option options[] = {
    {"query", required_argument, 0, 'i'},
    {"target", required_argument, 0, 'j'},
    {"gap-open", required_argument, 0, 'g'},
    {"gap-extend", required_argument, 0, 'e'},
    {"matrix", required_argument, 0, 'm'},
    {"cards", required_argument, 0, 'c'},
    {"out", required_argument, 0, 'o'},
    {"outfmt", required_argument, 0, 't'},
    {"algorithm", required_argument, 0, 'A'},
    {"cpu", no_argument, 0, 'P'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static CharInt outFormats[] = {
    { "pair", SW_OUT_PAIR },
    { "pair-stat", SW_OUT_STAT_PAIR },
    { "plot", SW_OUT_PLOT },
    { "stat", SW_OUT_STAT },
    { "dump", SW_OUT_DUMP }
};

static CharInt algorithms[] = {
    { "SW", SW_ALIGN },
    { "NW", NW_ALIGN },
    { "HW", HW_ALIGN },
    { "OV", OV_ALIGN }
};

static void getCudaCards(int** cards, int* cardsLen, char* optarg);

static int getOutFormat(char* optarg);
static int getAlgorithm(char* optarg);

static void help();

int main(int argc, char* argv[]) {

    char* queryPath = NULL;
    char* targetPath = NULL;

    char* matrix = "BLOSUM_62";
    
    int gapOpen = 10;
    int gapExtend = 1;
    
    int cardsLen = -1;
    int* cards = NULL;
    
    char* out = NULL;
    int outFormat = SW_OUT_STAT_PAIR;

    int algorithm = SW_ALIGN;

    int forceCpu = 0;

    while (1) {

        char argument = getopt_long(argc, argv, "i:j:g:e:m:h", options, NULL);

        if (argument == -1) {
            break;
        }

        switch (argument) {
        case 'i':
            queryPath = optarg;
            break;
        case 'j':
            targetPath = optarg;
            break;
        case 'g':
            gapOpen = atoi(optarg);
            break;
        case 'e':
            gapExtend = atoi(optarg);
            break;
        case 'm':
            matrix = optarg;
            break;
        case 'c':
            getCudaCards(&cards, &cardsLen, optarg);
            break;
        case 'o':
            out = optarg;
            break;
        case 't':
            outFormat = getOutFormat(optarg);
            break;
        case 'A':
            algorithm = getAlgorithm(optarg);
            break;
        case 'P':
            forceCpu = 1;
            break;
        case 'h':
        default:
            help();
            return -1;
        }
    }

    ASSERT(queryPath != NULL, "missing option -i (query file)");
    ASSERT(targetPath != NULL, "missing option -j (target file)");
    
    if (forceCpu) {
        cards = NULL;
        cardsLen = 0;
    } else {

        if (cardsLen == -1) {
            cudaGetCards(&cards, &cardsLen);
        }

        ASSERT(cudaCheckCards(cards, cardsLen), "invalid cuda cards");
    }
    
    Scorer* scorer;
    scorerCreateMatrix(&scorer, matrix, gapOpen, gapExtend);
    
    Chain* query = NULL;
    Chain* target = NULL; 
    
    readFastaChain(&query, queryPath);
    readFastaChain(&target, targetPath);

    threadPoolInitialize(cardsLen + 8);
    
    Alignment* alignment;
    alignPair(&alignment, algorithm, query, target, scorer, cards, cardsLen, NULL);
     
    ASSERT(checkAlignment(alignment), "invalid align");
    
    outputAlignment(alignment, out, outFormat);
    
    alignmentDelete(alignment);

    chainDelete(query);
    chainDelete(target);
    
    scorerDelete(scorer);
    
    threadPoolTerminate();
    free(cards);

    return 0;
}

static void getCudaCards(int** cards, int* cardsLen, char* optarg) {

    *cardsLen = strlen(optarg);
    *cards = (int*) malloc(*cardsLen * sizeof(int));
    
    int i;
    for (i = 0; i < *cardsLen; ++i) {
        (*cards)[i] = optarg[i] - '0';
    }
}

static int getOutFormat(char* optarg) {

    int i;
    for (i = 0; i < CHAR_INT_LEN(outFormats); ++i) {
        if (strcmp(outFormats[i].format, optarg) == 0) {
            return outFormats[i].code;
        }
    }

    ASSERT(0, "unknown out format %s", optarg);
}

static int getAlgorithm(char* optarg) {

    int i;
    for (i = 0; i < CHAR_INT_LEN(algorithms); ++i) {
        if (strcmp(algorithms[i].format, optarg) == 0) {
            return algorithms[i].code;
        }
    }

    ASSERT(0, "unknown algorithm %s", optarg);
}

static void help() {
    printf(
    "usage: swsharpp -i <query file> -j <target file> [arguments ...]\n"
    "\n"
    "arguments:\n"
    "    -i, --query <file>\n"
    "        (required)\n"
    "        input fasta query file\n"
    "    -j, --target <file>\n"
    "        (required)\n"
    "        input fasta target file\n"
    "    -g, --gap-open <int>\n"
    "        default: 10\n"
    "        gap opening penalty, must be given as a positive integer \n"
    "    -e, --gap-extend <int>\n"
    "        default: 1\n"
    "        gap extension penalty, must be given as a positive integer and\n"
    "        must be less or equal to gap opening penalty\n" 
    "    -m, --matrix <string>\n"
    "        default: BLOSUM_62\n"
    "        substitution matrix, can be one of the following: BLOSUM_45, BLOSUM_50,\n"
    "        BLOSUM_62, BLOSUM_80, BLOSUM_90, PAM_30, PAM_70, PAM_250\n"
    "    --algorithm <string>\n"
    "        default: SW\n"
    "        algorithm used for alignment, must be one of the following: \n"
    "            SW - Smith-Waterman local alignment\n"
    "            NW - Needleman-Wunsch global alignment\n"
    "            HW - semiglobal alignment\n"
    "            OV - overlap alignment\n"
    "    --cards <ints>\n"
    "        default: all available CUDA cards\n"
    "        list of cards should be given as an array of card indexes delimited with\n"
    "        nothing, for example usage of first two cards is given as --cards 01\n"
    "    --out <string>\n"
    "        default: stdout\n"
    "        output file for the alignment\n"
    "    --outfmt <string>\n"
    "        default: pair-stat\n"
    "        out format for the output file, must be one of the following: \n"
    "            pair      - emboss pair output format \n"
    "            pair-stat - combination of pair and stat output\n"
    "            plot      - output used for plotting alignment with gnuplot \n"
    "            stat      - statistics of the alignment\n"
    "            dump      - binary format for usage with swsharpout\n"
    "    --cpu\n"
    "        only cpu is used\n"
    "    -h, -help\n"
    "        prints out the help\n");
}
