#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    int i = (row * mat->cols) + col;
    return mat->data[i];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    int i = (row * mat->cols) + col;
    mat->data[i] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix *temp = malloc(sizeof(matrix));
    if (!temp) {
        return -2;
    }
    temp->rows = rows;
    temp->cols = cols;
    temp->ref_cnt = 1;
    temp->parent = NULL;
    temp->data = calloc(sizeof(double), rows * cols);
    if (!(temp->data)) {
        return -2;
    }
    *mat = temp;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (!mat) {
        return;
    } else if (!mat->parent) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix *temp = malloc(sizeof(matrix));
    if (!temp) {
        return -2;
    }
    temp->data = from->data + offset;
    temp->rows = rows;
    temp->cols = cols;
    temp->parent = from;
    temp->ref_cnt = 1;
    from->ref_cnt += 1;
    *mat = temp;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    __m256d vals = _mm256_set1_pd (val);
    int size = mat->rows * mat->cols;
    int blocks = size / 4;
    double *index = mat->data;
    #pragma omp parallel for
    for(int i = 0; i < blocks; i += 1) {
        _mm256_storeu_pd (index, vals);
        index += 4;
    }
    for(int i = blocks * 4; i < size; i += 1) {
        mat->data[i] = val;
    }   
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    __m256d mul = _mm256_set1_pd(-1);
    int size = mat->rows * mat->cols;
    int blocks = size / 4;
    double *index = mat->data;
    double *index2 = result->data;
    #pragma omp parallel
    {
        __m256d vec;
        __m256d neg;
        #pragma omp for
        for(int i = 0; i < blocks; i += 1) {
            int curr = 4 * i;
            vec = _mm256_loadu_pd(index + curr);
            neg = _mm256_mul_pd(mul, vec);
            vec = _mm256_max_pd(vec, neg);
            _mm256_storeu_pd(index2 + curr, vec);
        }
        for(int i = blocks * 4; i < size; i += 1) {
            int val = mat->data[i];
            if (val >= 0) {
                result->data[i] = val;
            } else {
                result->data[i] = 0 - val;
            }
        }
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int size = mat1->rows * mat1->cols;
    int blocks = size / 4;
    double *index1 = mat1->data;
    double *index2= mat2->data;
    double *index3 = result->data;
    #pragma omp parallel
    {
        __m256d vec1;
        __m256d vec2;
        __m256d sum_vec;
        #pragma omp for
        for(int i = 0; i < blocks; i += 1) {
            int curr = i * 4;
            vec1 = _mm256_loadu_pd(index1 + curr);
            vec2 = _mm256_loadu_pd(index2 + curr);
            sum_vec = _mm256_add_pd(vec1, vec2);
            _mm256_storeu_pd (index3 + curr, sum_vec);
        }
        for(int i = blocks * 4; i < size; i += 1) {
            result->data[i] = mat1->data[i] + mat2->data[i];
        }
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int rows1 = mat1->rows;
    int cols1 = mat1->cols;
    int rows2 = mat2->rows;
    int cols2 = mat2->cols;
    matrix *transpose;
    allocate_matrix(&transpose, cols2, rows2);
    transpose_matrix(transpose, mat2);
    int blocks = cols1 / 4;
    int big_blocks = cols1 / 16;
    #pragma omp parallel for
    for (int i = 0; i < rows1; i += 1) {
        for (int j = 0; j < cols2; j += 1) {
            __m256d vec1;
            __m256d vec2;
            double sum = 0;
            __m256d sum_vec = _mm256_set1_pd(0);
            double *index1 = (mat1->data) + cols1 * i;
            double *index2 = (transpose->data) + (cols1 * j);
            double *access = malloc(4 * sizeof(double));
            for (int k = 0; k < big_blocks; k += 1) {
                vec1 = _mm256_loadu_pd(index1);
                vec2 = _mm256_loadu_pd(index2);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec1, vec2));
                vec1 = _mm256_loadu_pd(index1 + 4);
                vec2 = _mm256_loadu_pd(index2 + 4);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec1, vec2));
                vec1 = _mm256_loadu_pd(index1 + 8);
                vec2 = _mm256_loadu_pd(index2 + 8);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec1, vec2));
                vec1 = _mm256_loadu_pd(index1 + 12);
                vec2 = _mm256_loadu_pd(index2 + 12);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec1, vec2));
                index1 += 16;
                index2 += 16;
            }
            for (int k = big_blocks * 4; k < blocks; k += 1) {
                vec1 = _mm256_loadu_pd(index1);
                vec2 = _mm256_loadu_pd(index2);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec1, vec2));
                index1 += 4;
                index2 += 4;
            }
            _mm256_storeu_pd(access, sum_vec);
            sum += access[0] + access[1] + access[2] + access[3];
            for (int k = blocks * 4; k < cols1; k += 1) {
                int index1 = (cols1 * i) + k;
                int index2 = (cols1 * j) + k;
                sum += mat1->data[index1] * transpose->data[index2];
            }
            int index = (cols2 * i) + j;
            result->data[index] = sum;
            free(access);
        }
    }
    deallocate_matrix(transpose);
    return 0;
}

int transpose_matrix(matrix *result, matrix *mat) {
    int rows = mat->rows;
    int cols = mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < rows; i += 1) {
        for (int j = 0; j < cols; j += 1) {
            result->data[rows * j + i] = mat->data[cols * i + j];
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
//Euler's
int pow_matrix(matrix *result, matrix *mat, int pow) {
    matrix *temp;
    matrix *res;
    allocate_matrix(&temp, mat->rows, mat->cols);
    allocate_matrix(&res, mat->rows, mat->cols);
    fill_matrix(result, 0);
    copy_matrix(temp, mat);
    for (int i = 0; i < result->rows; i += 1) {
        set(result, i, i, 1);
    }
    while(pow > 0) {
        if (pow % 2 == 0) {
            mul_matrix(res, temp, temp);
            pow /= 2;
            copy_matrix(temp, res);
        } else {
            mul_matrix(res, result, temp);
            pow -= 1;
            copy_matrix(result, res);
        }
    }
    return 0;
}

void copy_matrix(matrix *result, matrix *mat) {
    int rows = mat->rows;
    int cols = mat->cols;
    int blocks = cols / 4;
    #pragma omp parallel for
    for (int i = 0; i < rows; i += 1) {
        for (int j = 0; j < blocks; j += 1) {
            __m256d temp = _mm256_loadu_pd(mat->data + i * cols + j * 4);
            _mm256_storeu_pd(result->data + i * cols + j * 4, temp);
        }
        for (int j = blocks * 4; j < cols; j += 1) {
            set(result, i, j, get(mat, i, j));
        }
    }
}
