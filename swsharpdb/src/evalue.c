/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact the author by mkorpar@gmail.com.
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "swsharp/swsharp.h"

#include "evalue.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SCORER_CONSTANTS_LEN (sizeof(scorerConstants) / sizeof(ScorerConstants))

struct EValueParams {
    double lambda;
    double K;
    double H;
    double a;
    double C;
    double alpha;
    double sigma;
    double b;
    double beta;
    double tau;
    double G;
    double aUn;
    double alphaUn;
    int length;
};

typedef struct ScorerConstants {
    const char* matrix;
    int gapOpen;
    int gapExtend;
    double lambda;
    double K;
    double H;
    double a;
    double C;
    double alpha;
    double sigma;
} ScorerConstants;

// lambda, k, H, a, C, Alpha, Sigma
static ScorerConstants scorerConstants[] = {
    { "BLOSUM_62", -1, -1, 0.3176, 0.134, 0.4012, 0.7916, 0.623757, 4.964660, 4.964660 },
    { "BLOSUM_62", 11, 2, 0.297, 0.082, 0.27, 1.1, 0.641766, 12.673800, 12.757600 },
    { "BLOSUM_62", 10, 2, 0.291, 0.075, 0.23, 1.3, 0.649362, 16.474000, 16.602600 },
    { "BLOSUM_62", 9, 2, 0.279, 0.058, 0.19, 1.5, 0.659245, 22.751900, 22.950000 },
    { "BLOSUM_62", 8, 2, 0.264, 0.045, 0.15, 1.8, 0.672692, 35.483800, 35.821300 },
    { "BLOSUM_62", 7, 2, 0.239, 0.027, 0.10, 2.5, 0.702056, 61.238300, 61.886000 },
    { "BLOSUM_62", 6, 2, 0.201, 0.012, 0.061, 3.3, 0.740802, 140.417000, 141.882000 },
    { "BLOSUM_62", 13, 1, 0.292, 0.071, 0.23, 1.2, 0.647715, 19.506300, 19.893100 },
    { "BLOSUM_62", 12, 1, 0.283, 0.059, 0.19, 1.5, 0.656391, 27.856200, 28.469900 },
    { "BLOSUM_62", 11, 1, 0.267, 0.041, 0.14, 1.9, 0.669720, 42.602800, 43.636200 },
    { "BLOSUM_62", 10, 1, 0.243, 0.024, 0.10, 2.5, 0.693267, 83.178700, 85.065600 },
    { "BLOSUM_62", 9, 1, 0.206, 0.010, 0.052, 4.0, 0.731887, 210.333000, 214.842000 },
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static double calculateEValue(int score, int queryLen, int targetLen, 
    EValueParams* params);
    
#ifdef _WIN32
double erf(double x);
#endif

//******************************************************************************

//******************************************************************************
// PUBLIC

extern EValueParams* createEValueParams(Chain** database, int databaseLen, 
    Scorer* scorer) {

    int i;
    
    int length = 0;
    for (i = 0; i < databaseLen; ++i) {
        length += chainGetLength(database[i]);
    }
    
    const char* matrix = scorerGetName(scorer);
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    
    double alphaUn = scorerConstants[0].alpha;
    double aUn = scorerConstants[0].a;
    double G = gapOpen + gapExtend;
    
    int index = -1;
    for (i = 0; i < SCORER_CONSTANTS_LEN; ++i) {

        ScorerConstants entry = scorerConstants[i];
        
        if (entry.gapOpen == gapOpen && entry.gapExtend == gapExtend &&
            strncmp(entry.matrix, matrix, strlen(entry.matrix)) == 0) {
            index = i;
            break;
        }
    }
    
    if (index == -1) {
        index = 0;
        printf("WARNING: no e-value params found, using defaults\n");
    }
    
    EValueParams* params = (EValueParams*) malloc(sizeof(struct EValueParams));
    
    params->G = G;
    params->aUn = aUn;
    params->alphaUn = alphaUn;
    params->lambda = scorerConstants[index].lambda;
    params->K = scorerConstants[index].K;
    params->H = scorerConstants[index].H;
    params->a = scorerConstants[index].a;
    params->C = scorerConstants[index].C;
    params->alpha = scorerConstants[index].alpha;
    params->sigma = scorerConstants[index].sigma;
    params->b = 2.0 * G * (params->aUn - params->a);
    params->beta = 2.0 * G * (params->alphaUn - params->alpha);
    params->tau = 2.0 * G * (params->alphaUn - params->sigma);
    params->length = length;
    
    return params;
}

extern void deleteEValueParams(EValueParams* eValueParams) {
    free(eValueParams);
    eValueParams = NULL;
}

extern void eValues(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, EValueParams* eValueParams) {
    
    int queryLen = chainGetLength(query);

    int i;
    for (i = 0; i < databaseLen; ++i) {
        
        int score = scores[i];
        int targetLen = chainGetLength(database[i]);

        if (score == NO_SCORE) {
            values[i] = INFINITY;
            continue;
        }
        
        values[i] = calculateEValue(score, queryLen, targetLen, eValueParams);
    }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

static double calculateEValue(int score, int queryLen, int targetLen, 
    EValueParams* params) {
    
    // code taken from blast
    // pile of statistical crap

    int y_ = score;
    int m_ = queryLen;
    int n_ = targetLen;
    
    // the pair-wise e-value must be scaled back to db-wise e-value
    double db_scale_factor = (double) params->length / (double) n_;

    double lambda_    = params->lambda;
    double k_         = params->K;
    double ai_hat_    = params->a;
    double bi_hat_    = params->b;
    double alphai_hat_= params->alpha;
    double betai_hat_ = params->beta;
    double sigma_hat_ = params->sigma;
    double tau_hat_   = params->tau;

    // here we consider symmetric matrix only
    double aj_hat_    = ai_hat_;
    double bj_hat_    = bi_hat_;
    double alphaj_hat_= alphai_hat_;
    double betaj_hat_ = betai_hat_;

    // this is 1/sqrt(2.0*PI)
    static double const_val = 0.39894228040143267793994605993438;
    double m_li_y, vi_y, sqrt_vi_y, m_F, P_m_F;
    double n_lj_y, vj_y, sqrt_vj_y, n_F, P_n_F;
    double c_y, p1, p2;
    double area;

    m_li_y = m_ - (ai_hat_*y_ + bi_hat_);
    vi_y = MAX(2.0*alphai_hat_/lambda_, alphai_hat_*y_+betai_hat_);
    sqrt_vi_y = sqrt(vi_y);
    m_F = m_li_y/sqrt_vi_y;
    P_m_F = 0.5 + 0.5 * erf(m_F);
    p1 = m_li_y * P_m_F + sqrt_vi_y * const_val * exp(-0.5*m_F*m_F);

    n_lj_y = n_ - (aj_hat_*y_ + bj_hat_);
    vj_y = MAX(2.0*alphaj_hat_/lambda_, alphaj_hat_*y_+betaj_hat_);
    sqrt_vj_y = sqrt(vj_y);
    n_F = n_lj_y/sqrt_vj_y;
    P_n_F = 0.5 + 0.5 * erf(n_F);
    p2 = n_lj_y * P_n_F + sqrt_vj_y * const_val * exp(-0.5*n_F*n_F);

    c_y = MAX(2.0*sigma_hat_/lambda_, sigma_hat_*y_+tau_hat_);
    area = p1 * p2 + c_y * P_m_F * P_n_F;

    return area * k_ * exp(-lambda_ * y_) * db_scale_factor;
}

#ifdef _WIN32
double erf(double x) {

    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = x < 0 ? -1 : 1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign * y;
}
#endif

//******************************************************************************
