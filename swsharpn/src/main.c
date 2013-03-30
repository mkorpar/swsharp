#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "swsharp/swsharp.h"

static struct option options[] = {
    {"query", required_argument, 0, 'i'},
    {"target", required_argument, 0, 'j'},
    {"gap-open", required_argument, 0, 'g'},
    {"gap-extend", required_argument, 0, 'e'},
    {"match", required_argument, 0, 'a'},
    {"mismatch", required_argument, 0, 'b'},
    {"cards", required_argument, 0, 'c'},
    {"out", required_argument, 0, 'o'},
    {"dump", required_argument, 0, 'd'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};

static void help();

int main(int argc, char* argv[]) {

    char* queryPath = NULL;
    char* targetPath = NULL;

    int match = 1;
    int mismatch = -3;
    
    int gapOpen = 5;
    int gapExtend = 2;
    
    int cardsLen = -1;
    int* cards = NULL;
    
    char* out = NULL;
    char* dump = NULL;
    
    int i;
    
    while (1) {

        char argument = getopt_long(argc, argv, "i:j:g:e:h", options, NULL);

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
        case 'a':
            match = atoi(optarg);
            break;
        case 'b':
            mismatch = atoi(optarg);
            break;
        case 'c':
            cardsLen = strlen(optarg);
            for (i = 0; i < cardsLen; ++i) cards[i] = optarg[i] - '0';
            break;
        case 'o':
            out = optarg;
            break;
        case 'd':
            dump = optarg;
            break;
        case 'h':
        default:
            help();
            return -1;
        }
    }

    ASSERT(queryPath != NULL, "missing option -i (query file)");
    ASSERT(targetPath != NULL, "missing option -j (target file)");
    
    if (cardsLen == -1) {
        cudaGetCards(&cards, &cardsLen);
    }
    
    ASSERT(cudaCheckCards(cards, cardsLen), "invalid cuda cards");
    
    ASSERT(match > 0, "invalid match");
    ASSERT(mismatch < 0, "invalid mismatch");
    
    ASSERT(gapExtend > 0 && gapExtend <= gapOpen, "invalid gap extend");
    
    Scorer* scorer;
    scorerCreateConst(&scorer, match, mismatch, gapOpen, gapExtend);
    
    Chain* query = NULL;
    Chain* target = NULL; 
    
    readFastaChain(&query, queryPath);
    readFastaChain(&target, targetPath);

    Alignment* alignment;
    alignPair(&alignment, query, target, scorer, SW_ALIGN, cards, cardsLen, NULL);
     
    ASSERT(checkAlignment(alignment), "invalid align");
    
    outputPair(alignment, out);
        
    if (dump != NULL) {
        dumpAlignment(alignment, dump);
    }
    
    alignmentDelete(alignment);

    chainDelete(query);
    chainDelete(target);
    
    scorerDelete(scorer);
    
    free(cards);

    return 0;
}

static void help() {
    printf("bok\n");
}
