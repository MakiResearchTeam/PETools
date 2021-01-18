#include <iostream>
#include <algorithm>
#include <math.h>
#include "pafprocess.h"

#define PEAKS(i, j) peak_info_data[j+p2*i]
#define PAF(i, j, k) pafmap[k+f3*(j+f2*i)]

using namespace std;

vector<vector<float> > subset;
vector<Peak> peak_infos_line;

int roundpaf(float v);
vector<VectorXY> get_paf_vectors(float *pafmap, const int& ch_id1, const int& ch_id2, int& f2, int& f3, Peak& peak1, Peak& peak2);
bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b);

int process_paf(
    int size_score_from_peaks, float *score_from_peaks,
    int p1, int p2, int *peak_info_data,
    int f1, int f2, int f3, float *pafmap
){
    peak_infos_line.clear();
    vector<Peak> peak_infos[NUM_PART];

    for (int peaks_indx = 0; peaks_indx < p1; peaks_indx++){
        // peak_info_data - first 2 dimensions - y, x coord
        // last dimension - keypoint class

        Peak info;
        info.id = peaks_indx;
        info.y = PEAKS(peaks_indx, 0);
        info.x = PEAKS(peaks_indx, 1);
        info.score = score_from_peaks[peaks_indx];

        peak_infos[PEAKS(peaks_indx, 2)].push_back(info);
        peak_infos_line.push_back(info);
    }

    // Start to Connect
    vector<Connection> connection_all[COCOPAIRS_SIZE];
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id ++) {
        vector<ConnectionCandidate> candidates;
        vector<Peak>& peak_a_list = peak_infos[COCOPAIRS[pair_id][0]];
        vector<Peak>& peak_b_list = peak_infos[COCOPAIRS[pair_id][1]];

        if (peak_a_list.size() == 0 || peak_b_list.size() == 0) {
            continue;
        }

        for (int peak_a_id = 0; peak_a_id < (int) peak_a_list.size(); peak_a_id ++) {
            Peak& peak_a = peak_a_list[peak_a_id];
            for (int peak_b_id = 0; peak_b_id < (int) peak_b_list.size(); peak_b_id ++) {
                Peak& peak_b = peak_b_list[peak_b_id];

                // calculate vector(direction)
                VectorXY vec;
                vec.x = peak_b.x - peak_a.x;
                vec.y = peak_b.y - peak_a.y;
                float norm = (float) sqrt(vec.x * vec.x + vec.y * vec.y);
                if (norm < 1e-6) continue;
                vec.x = vec.x / norm;
                vec.y = vec.y / norm;

                vector<VectorXY> paf_vecs = get_paf_vectors(pafmap, COCOPAIRS_NET[pair_id][0], COCOPAIRS_NET[pair_id][1], f2, f3, peak_a, peak_b);
                float scores = 0.0f;

                // criterion 1 : score treshold count
                int criterion1 = 0;
                for (int i = 0; i < STEP_PAF; i ++) {
                    float score = vec.x * paf_vecs[i].x + vec.y * paf_vecs[i].y;
                    scores += score;

                    if (score > THRESH_VECTOR_SCORE) criterion1 += 1;
                }

                float criterion2 = scores / STEP_PAF + min(0.0, 0.5 * f1 / norm - 1.0);

                if (criterion1 > THRESH_VECTOR_CNT1 && criterion2 > 0) {
                    ConnectionCandidate candidate;
                    candidate.idx1 = peak_a_id;
                    candidate.idx2 = peak_b_id;
                    candidate.score = criterion2;
                    candidate.etc = criterion2 + peak_a.score + peak_b.score;
                    candidates.push_back(candidate);
                }
            }
        }

        vector<Connection>& conns = connection_all[pair_id];
        sort(candidates.begin(), candidates.end(), comp_candidate);
        for (int c_id = 0; c_id < (int) candidates.size(); c_id ++) {
            ConnectionCandidate& candidate = candidates[c_id];
            bool assigned = false;
            for (int conn_id = 0; conn_id < (int) conns.size(); conn_id ++) {
                if (conns[conn_id].peak_id1 == candidate.idx1) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
                if (conns[conn_id].peak_id2 == candidate.idx2) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
            }
            if (assigned) continue;

            Connection conn;
            conn.peak_id1 = candidate.idx1;
            conn.peak_id2 = candidate.idx2;
            conn.score = candidate.score;
            conn.cid1 = peak_a_list[candidate.idx1].id;
            conn.cid2 = peak_b_list[candidate.idx2].id;
            conns.push_back(conn);
        }
    }

    // Generate subset
    /*
    Last two indexes is used for calculating detected limb
    And all limbs that appear on the image
    `row` itself store keypoints, same as process in this code, it work only with keypoints!
    */
    subset.clear();
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id ++) {
        vector<Connection>& conns = connection_all[pair_id];
        int part_id1 = COCOPAIRS[pair_id][0];
        int part_id2 = COCOPAIRS[pair_id][1];

        for (int conn_id = 0; conn_id < (int) conns.size(); conn_id ++) {
            int found = 0;
            int subset_idx1 = 0, subset_idx2 = 0;
            for (int subset_id = 0; subset_id < (int) subset.size(); subset_id ++) {
                if (subset[subset_id][part_id1] == conns[conn_id].cid1 || subset[subset_id][part_id2] == conns[conn_id].cid2) {
                    if (found == 0) subset_idx1 = subset_id;
                    if (found == 1) subset_idx2 = subset_id;
                    found += 1;
                }
            }

            if (found == 1) {
                if (subset[subset_idx1][part_id2] != conns[conn_id].cid2) {
                    subset[subset_idx1][part_id2] = conns[conn_id].cid2;
                    subset[subset_idx1][SIZE_ROW - 1] += 1;
                    subset[subset_idx1][SIZE_ROW - 2] += peak_infos_line[conns[conn_id].cid2].score + conns[conn_id].score;
                }
            } else if (found == 2) {
                int membership = 0;
                for (int subset_id = 0; subset_id < SIZE_ROW - 2 && membership != 2; subset_id ++) {
                    if (subset[subset_idx1][subset_id] > 0 && subset[subset_idx2][subset_id] > 0) {
                        membership = 2;
                    }
                }

                if (membership == 0) {
                    for (int subset_id = 0; subset_id < SIZE_ROW - 2; subset_id ++) subset[subset_idx1][subset_id] += (subset[subset_idx2][subset_id] + 1);

                    subset[subset_idx1][SIZE_ROW - 1] += subset[subset_idx2][SIZE_ROW - 1];
                    subset[subset_idx1][SIZE_ROW - 2] += subset[subset_idx2][SIZE_ROW - 2];
                    subset[subset_idx1][SIZE_ROW - 2] += conns[conn_id].score;
                    subset.erase(subset.begin() + subset_idx2);
                } else {
                    subset[subset_idx1][part_id2] = conns[conn_id].cid2;
                    subset[subset_idx1][SIZE_ROW - 1] += 1;
                    subset[subset_idx1][SIZE_ROW - 2] += peak_infos_line[conns[conn_id].cid2].score + conns[conn_id].score;
                }
            // If found zero and pair id that should be drawn
            // Some limbs were added just for additional help for paff to connect other points
            } else if (found == 0 && pair_id < NUM_PART) {
                // last two dimension hold for detected limbs and score of all limbs
                vector<float> row(SIZE_ROW);
                for (int i = 0; i < SIZE_ROW; i ++) row[i] = -1;
                row[part_id1] = conns[conn_id].cid1;
                row[part_id2] = conns[conn_id].cid2;
                // at this moments we have 2 detected limbs
                row[SIZE_ROW - 1] = 2;
                // these limb have next score
                row[SIZE_ROW - 2] = peak_infos_line[conns[conn_id].cid1].score +
                          peak_infos_line[conns[conn_id].cid2].score +
                          conns[conn_id].score;
                subset.push_back(row);
            }
        }
    }

    // Delete some rows in they thresh hold/score is lower that specified in constants
    for (int i = subset.size() - 1; i >= 0; i --) {
        if (subset[i][SIZE_ROW - 1] < THRESH_PART_CNT || subset[i][SIZE_ROW - 2] / subset[i][SIZE_ROW - 1] < THRESH_HUMAN_SCORE)
            subset.erase(subset.begin() + i);
    }

    return 0;
}

int get_num_humans() {
    return subset.size();
}

int get_part_cid(int human_id, int part_id) {
    return subset[human_id][part_id];
}

float get_score(int human_id) {
    return subset[human_id][COCOPAIRS_SIZE - 2] / subset[human_id][COCOPAIRS_SIZE - 1];
}

int get_part_x(int cid) {
    return peak_infos_line[cid].x;
}
int get_part_y(int cid) {
    return peak_infos_line[cid].y;
}
float get_part_score(int cid) {
    return peak_infos_line[cid].score;
}

vector<VectorXY> get_paf_vectors(float *pafmap, const int& ch_id1, const int& ch_id2, int& f2, int& f3, Peak& peak1, Peak& peak2) {
    vector<VectorXY> paf_vectors;

    const float STEP_X = (peak2.x - peak1.x) / float(STEP_PAF);
    const float STEP_Y = (peak2.y - peak1.y) / float(STEP_PAF);

    for (int i = 0; i < STEP_PAF; i ++) {
        int location_x = roundpaf(peak1.x + i * STEP_X);
        int location_y = roundpaf(peak1.y + i * STEP_Y);

        VectorXY v;
        v.x = PAF(location_y, location_x, ch_id1);
        v.y = PAF(location_y, location_x, ch_id2);
        paf_vectors.push_back(v);
    }

    return paf_vectors;
}

int roundpaf(float v) {
    return (int) (v + 0.5);
}

bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b) {
    return a.score > b.score;
}
