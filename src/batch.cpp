#include "batch.hpp"
#include "tensors.hpp" // for views/blas

template<typename P>
batch_list<P>::batch_list(int const num_batch, int const nrows, int const ncols,
                          int const stride)
    : num_batch(num_batch), nrows(nrows), ncols(ncols),
      stride(stride), batch_list_{new P *[num_batch]()}
{
  assert(num_batch > 0);
  assert(nrows > 0);
  assert(ncols > 0);
  assert(stride >= nrows);

  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> const &other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride), batch_list_{new P *[other.num_batch]()}
{
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> const &other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
  return *this;
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> &&other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride), batch_list_{other.batch_list_}
{
  other.batch_list_ = nullptr;
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> &&other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);

  batch_list_       = other.batch_list_;
  other.batch_list_ = nullptr;
  return *this;
}

template<typename P>
batch_list<P>::~batch_list()
{
  delete[] batch_list_;
}

template<typename P>
bool batch_list<P>::operator==(batch_list<P> other) const
{
  if (nrows != other.nrows)
  {
    return false;
  }
  if (ncols != other.ncols)
  {
    return false;
  }
  if (stride != other.stride)
  {
    return false;
  }
  if (num_batch != other.num_batch)
  {
    return false;
  }

  for (int i = 0; i < num_batch; ++i)
  {
    if (batch_list_[i] != other.batch_list_[i])
    {
      return false;
    }
  }

  return true;
}

template<typename P>
P *batch_list<P>::operator()(int const position) const
{
  assert(position >= 0);
  assert(position < num_batch);
  return batch_list_[position];
}

// insert the provided view's data pointer
// at the index indicated by position argument
// cannot overwrite previous assignment
template<typename P>
void batch_list<P>::insert(fk::matrix<P, mem_type::view> const a,
                           int const position)
{
  // make sure this matrix is the
  // same dimensions as others in batch
  assert(a.nrows() == nrows);
  assert(a.ncols() == ncols);
  assert(a.stride() == stride);

  // ensure position is valid
  assert(position >= 0);
  assert(position < num_batch);

  // ensure nothing already assigned
  assert(!batch_list_[position]);

  batch_list_[position] = a.data();
}

// clear one assignment
// returns true if there was a previous assignment,
// false if nothing was assigned
template<typename P>
bool batch_list<P>::clear(int const position)
{
  P *temp               = batch_list_[position];
  batch_list_[position] = nullptr;
  return temp;
}

// get a pointer to the batch_list's
// pointers for batched blas call
// for performance, may have to
// provide a direct access to P**
// from batch_list_, but avoid for now
template<typename P>
P **batch_list<P>::get_list() const
{
  P **const list_copy = new P *[num_batch]();
  std::memcpy(list_copy, batch_list_, num_batch * sizeof(P *));
  return list_copy;
}

// verify that every allocated pointer
// has been assigned to
template<typename P>
bool batch_list<P>::is_filled() const
{
  for (P *const ptr : (*this))
  {
    if (!ptr)
    {
      return false;
    }
  }
  return true;
}

// clear assignments
template<typename P>
batch_list<P> &batch_list<P>::clear_all()
{
  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
  return *this;
}

// execute a batched gemm given a, b, c batch lists
// and other blas information
template<typename P>
void batchedGemm(batch_list<P> const a, batch_list<P> const b,
                 batch_list<P> const c, P alpha, P beta, bool trans_a,
                 bool trans_b)
{
  // check data validity
  assert(a.is_filled() && b.is_filled() && c.is_filled());

  // check cardinality of sets
  assert(a.num_batch == b.num_batch);
  assert(b.num_batch == c.num_batch);
  int const num_batch = a.num_batch;

  // check dimensions for gemm
  int const rows_a = trans_a ? a.ncols : a.nrows;
  int const cols_a = trans_a ? a.nrows : a.ncols;
  int const rows_b = trans_b ? b.ncols : b.nrows;
  int const cols_b = trans_b ? b.nrows : b.ncols;
  assert(cols_a == rows_b);
  assert(c.nrows == rows_a);
  assert(c.ncols == cols_b);

  // setup blas args
  int m                  = rows_a;
  int n                  = rows_b;
  int k                  = cols_a;
  int lda                = a.stride;
  int ldb                = b.stride;
  int ldc                = c.stride;
  char const transpose_a = trans_a ? 't' : 'n';
  char const transpose_b = trans_b ? 't' : 'n';

  if constexpr (std::is_same<P, double>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::dgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha, a(i), &lda,
                 b(i), &ldb, &beta, c(i), &ldc);
    }
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::sgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha, a(i), &lda,
                 b(i), &ldb, &beta, c(i), &ldc);
    }
  }
}

template class batch_list<float>;
template class batch_list<double>;

template void batchedGemm(batch_list<float> const a, batch_list<float> const b,
                          batch_list<float> const c, float alpha, float beta,
                          bool trans_a, bool trans_b);

template void batchedGemm(batch_list<double> const a,
                          batch_list<double> const b,
                          batch_list<double> const c, double alpha, double beta,
                          bool trans_a, bool trans_b);
