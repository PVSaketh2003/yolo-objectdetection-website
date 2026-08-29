#ifndef LAPJV_HPP
#define LAPJV_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * Jonker-Volgenant (LAPJV) Linear Assignment Problem Solver
 * Solves the minimum cost bipartite matching problem between rows (tracks) and cols (detections).
 * Exception-Safe & Guaranteed Bounds Check Implementation.
 */
class LAPJV {
public:
    static std::vector<int> solve(const std::vector<std::vector<float>>& cost_matrix, float max_cost = 0.65f) {
        int n_rows = static_cast<int>(cost_matrix.size());
        if (n_rows == 0) return {};

        int n_cols = static_cast<int>(cost_matrix[0].size());
        if (n_cols == 0) return std::vector<int>(n_rows, -1);

        int dim = std::max(n_rows, n_cols);
        float LARGE_COST = 1e6f;

        std::vector<std::vector<float>> assign_cost(dim, std::vector<float>(dim, LARGE_COST));
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                assign_cost[r][c] = cost_matrix[r][c];
            }
        }

        std::vector<float> u(dim, 0.0f);
        std::vector<float> v(dim, 0.0f);
        std::vector<int> col_sol(dim, -1);
        std::vector<int> row_sol(dim, -1);

        for (int i = 0; i < dim; ++i) {
            std::vector<int> links(dim, -1);
            std::vector<float> mins(dim, LARGE_COST);
            std::vector<bool> visited(dim, false);

            int marked_row = i;
            int marked_col = -1;
            int j = -1;
            int iter_guard = 0;

            while (marked_row != -1 && iter_guard++ < dim * 2) {
                j = -1;
                float min_val = LARGE_COST;

                for (int j_candidate = 0; j_candidate < dim; ++j_candidate) {
                    if (!visited[j_candidate]) {
                        float cur_cost = assign_cost[marked_row][j_candidate] - u[marked_row] - v[j_candidate];
                        if (cur_cost < mins[j_candidate]) {
                            mins[j_candidate] = cur_cost;
                            links[j_candidate] = marked_col;
                        }
                        if (mins[j_candidate] < min_val) {
                            min_val = mins[j_candidate];
                            j = j_candidate;
                        }
                    }
                }

                if (j == -1 || j < 0 || j >= dim) break;

                for (int j_candidate = 0; j_candidate < dim; ++j_candidate) {
                    if (visited[j_candidate]) {
                        if (col_sol[j_candidate] >= 0 && col_sol[j_candidate] < dim) {
                            u[col_sol[j_candidate]] += min_val;
                        }
                        v[j_candidate] -= min_val;
                    } else {
                        mins[j_candidate] -= min_val;
                    }
                }
                u[i] += min_val;

                visited[j] = true;
                marked_col = j;
                marked_row = col_sol[j];
            }

            int current_j = j;
            int path_guard = 0;
            while (current_j >= 0 && current_j < dim && links[current_j] >= 0 && links[current_j] < dim && path_guard++ < dim) {
                col_sol[current_j] = col_sol[links[current_j]];
                current_j = links[current_j];
            }
            if (current_j >= 0 && current_j < dim) {
                col_sol[current_j] = i;
            }
        }

        for (int j = 0; j < dim; ++j) {
            if (col_sol[j] >= 0 && col_sol[j] < dim) {
                row_sol[col_sol[j]] = j;
            }
        }

        std::vector<int> result(n_rows, -1);
        for (int r = 0; r < n_rows; ++r) {
            int c = row_sol[r];
            if (c >= 0 && c < n_cols) {
                if (cost_matrix[r][c] <= max_cost) {
                    result[r] = c;
                }
            }
        }

        return result;
    }
};

#endif // LAPJV_HPP
