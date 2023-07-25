#pragma once

#include "elements.hpp"

namespace asgard
{
/*!
 * \brief Keeps track of the connectivity of the elements in the 1d hierarchy.
 *
 * Constructs a sparse matrix-like structure with row-compressed format and
 * ordered indexes within each row, so that the 1D connectivity can be verified
 * with a simple binary search.
 *
 * The advantage of the structure is to provide:
 * - easier check if two 1D cell are connected or not
 * - index of the connection, so operator coefficient matrices can be easily
 *   referenced
 */
class connect_1d
{
public:
  /*!
   *  \brief Constructor, makes the connectivity up to and including the given
   *         max-level.
   */
  connect_1d(int const max_level)
      : levels(max_level), cells(1 << levels), pntr(cells + 1, 0),
        indx(2 * cells)
  {
    std::vector<int> cell_per_level(levels + 2, 1);
    for (int l = 2; l < levels + 2; l++)
      cell_per_level[l] = 2 * cell_per_level[l - 1];

    // first two cells are connected to everything
    pntr[1] = cells;
    pntr[2] = 2 * cells;
    for (int i = 0; i < cells; i++)
      indx[i] = i;
    for (int i = 0; i < cells; i++)
      indx[i + cells] = i;

    // for the remaining, loop level by level, cell by cell
    for (int l = 2; l < levels + 1; l++)
    {
      int level_size = cell_per_level[l]; // number of cells in this level

      // for each cell in this level, look at all cells connected
      // look at previous levels, this level, follow on levels

      // start with the first cell, on the left edge
      int i = level_size; // index of the first cell
      // always connected to cells 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      // look at cells above
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // look at this level
      indx.push_back(i);
      indx.push_back(i + 1);
      // connect also to the right-most cell (periodic boundary)
      if (l > 2) // at level l = 2, i+1 is the right-most cell
        indx.push_back(cell_per_level[l + 1] - 1);
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // connect to the first bunch of cell, i.e., with overlapping support
        // going on by 2, 4, 8 ... and one more for touching boundary
        // also connect to the right-most cell
        int lstart = cell_per_level[downl];
        for (int downp = 0; downp < cell_per_level[downl - l + 1] + 1; downp++)
          indx.push_back(lstart + downp);
        indx.push_back(cell_per_level[downl + 1] - 1);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with point

      // handle middle cells
      for (int p = 1; p < level_size - 1; p++)
      {
        i++;
        // always connected to the first two cells
        indx.push_back(0);
        indx.push_back(1);
        // ancestors on previous levels
        for (int upl = 2; upl < l; upl++)
        {
          int segment_size = cell_per_level[l - upl + 1];
          int ancestor     = p / segment_size;
          int edge         = p - ancestor * segment_size; // p % segment_size
          // if on the left edge of the ancestor
          if (edge == 0)
            indx.push_back(cell_per_level[upl] + ancestor - 1);
          indx.push_back(cell_per_level[upl] + ancestor);
          // if on the right edge of the ancestor
          if (edge == segment_size - 1)
            indx.push_back(cell_per_level[upl] + ancestor + 1);
        }
        // on this level
        indx.push_back(i - 1);
        indx.push_back(i);
        indx.push_back(i + 1);
        // kids on further levels
        int left_kid = p; // initialize, will be updated on first iteration
        int num_kids = 1;
        for (int downl = l + 1; downl < levels + 1; downl++)
        {
          left_kid *= 2;
          num_kids *= 2;
          for (int j = left_kid - 1; j < left_kid + num_kids + 1; j++)
            indx.push_back(cell_per_level[downl] + j);
        }
        pntr[i + 1] = static_cast<int>(indx.size()); // done with cell i
      }

      // right edge cell
      i++;
      // always connected to 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // at this level
      // connect also to the left-most cell (periodic boundary)
      if (l > 2) // at level l = 2, left-most cell is i-1, don't double add
        indx.push_back(cell_per_level[l]);
      indx.push_back(i - 1);
      indx.push_back(i);
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // left edge on the level
        indx.push_back(cell_per_level[downl]);
        // get the last bunch of cells at the level
        int lend = cell_per_level[downl + 1] - 1;
        for (int downp = cell_per_level[downl - l + 1]; downp > -1; downp--)
          indx.push_back(lend - downp);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with the right edge
    } // done with level, move to the next level
  }   // close the constructor

  int get_offset(int row, int col) const
  {
    // first two levels are large and trivial, no need to search
    if (row == 0)
      return col;
    else if (row == 1)
      return cells + col;
    // if not on the first or second row, do binary search
    int sstart = pntr[row], send = pntr[row + 1] - 1;
    int current = (sstart + send) / 2;
    while (sstart <= send)
    {
      if (indx[current] < col)
      {
        sstart = current + 1;
      }
      else if (indx[current] > col)
      {
        send = current - 1;
      }
      else
      {
        return current;
      };
      current = (sstart + send) / 2;
    }
    return -1;
  }

  int num_connections() const { return static_cast<int>(indx.size()); }

  int num_cells() const { return cells; }

  int row_begin(int row) const { return pntr[row]; }

  int row_end(int row) const { return pntr[row + 1]; }

  int operator[](int j) const { return indx[j]; }

  int max_loaded_level() const { return levels; }

private:
  int levels;
  int cells;
  std::vector<int> pntr;
  std::vector<int> indx;
};

} // namespace asgard