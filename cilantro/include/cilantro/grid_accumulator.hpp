#pragma once

#include <map>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, ptrdiff_t EigenCoeff>
    struct EigenVectorComparatorHelper {
        enum { coeff = EigenDim - EigenCoeff };

        static inline bool result(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT,EigenDim,1> &p2) {
            if (p1[coeff] < p2[coeff]) return true;
            if (p2[coeff] < p1[coeff]) return false;
            return EigenVectorComparatorHelper<ScalarT,EigenDim,EigenCoeff-1>::result(p1, p2);
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct EigenVectorComparatorHelper<ScalarT, EigenDim, 1> {
        enum { coeff = EigenDim - 1 };

        static inline bool result(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT,EigenDim,1> &p2) {
            return p1[coeff] < p2[coeff];
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct EigenVectorComparator {
        inline bool operator()(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT,EigenDim,1> &p2) const {
            return EigenVectorComparatorHelper<ScalarT,EigenDim,EigenDim>::result(p1, p2);
        }
    };

    template <typename ScalarT>
    struct EigenVectorComparator<ScalarT, Eigen::Dynamic> {
        inline bool operator()(const Eigen::Matrix<ScalarT,Eigen::Dynamic,1> &p1, const Eigen::Matrix<ScalarT,Eigen::Dynamic,1> &p2) const {
            for (size_t i = 0; i < p1.rows() - 1; i++) {
                if (p1[i] < p2[i]) return true;
                if (p2[i] < p1[i]) return false;
            }
            return p1[p1.rows()-1] < p2[p1.rows()-1];
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim, class AccumulatorProxy>
    class GridAccumulator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        typedef typename AccumulatorProxy::Accumulator Accumulator;

        typedef Eigen::Matrix<ptrdiff_t,EigenDim,1> GridPoint;

        typedef typename std::conditional<(EigenDim != Eigen::Dynamic && sizeof(GridPoint) % 16 == 0) || (Accumulator::EigenAlign > 0),
                std::map<GridPoint,Accumulator,EigenVectorComparator<typename GridPoint::Scalar,EigenDim>,Eigen::aligned_allocator<std::pair<const GridPoint,Accumulator>>>,
                std::map<GridPoint,Accumulator,EigenVectorComparator<typename GridPoint::Scalar,EigenDim>>>::type GridBinMap;

        GridAccumulator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data,
                        const Eigen::Ref<const Vector<ScalarT,EigenDim>> &bin_size,
                        const AccumulatorProxy &accum_proxy)
                : data_map_(data),
                  bin_size_(bin_size),
                  bin_size_inv_(bin_size_.cwiseInverse())
        {
            build_index_(accum_proxy);
        }

        GridAccumulator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data,
                        ScalarT bin_size,
                        const AccumulatorProxy &accum_proxy)
                : data_map_(data),
                  bin_size_(Vector<ScalarT,EigenDim>::Constant(data_map_.rows(), 1, bin_size)),
                  bin_size_inv_(bin_size_.cwiseInverse())
        {
            build_index_(accum_proxy);
        }

        ~GridAccumulator() {}

        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getPointsMatrixMap() const { return data_map_; }

        inline const Vector<ScalarT,EigenDim>& getBinSize() const { return bin_size_; }

        inline const GridBinMap& getOccupiedBinMap() const { return grid_lookup_table_; }

        inline const std::vector<typename GridBinMap::iterator>& getOccupiedBinIterators() const { return bin_iterators_; }

        const typename GridBinMap::const_iterator findContainingGridBin(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point) const {
            GridPoint grid_coords(data_map_.rows());
            for (size_t i = 0; i < data_map_.rows(); i++) {
                grid_coords[i] = std::floor(point[i]*bin_size_inv_[i]);
            }
            return grid_lookup_table_.find(grid_coords);
        }

        inline const typename GridBinMap::const_iterator findContainingGridBin(size_t ind) const {
            return getPointBinNeighbors(data_map_.col(ind));
        }

        inline GridPoint getPointGridCoordinates(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point) const {
            GridPoint grid_coords(data_map_.rows());
            for (size_t i = 0; i < data_map_.rows(); i++) {
                grid_coords[i] = std::floor(point[i]*bin_size_inv_[i]);
            }
            return grid_coords;
        }

        inline GridPoint getPointGridCoordinates(size_t point_ind) const {
            return getGridCoordinates(data_map_.col(point_ind));
        }

        inline Vector<ScalarT,EigenDim> getBinCornerCoordinates(const Eigen::Ref<const GridPoint> &grid_point) const {
            Vector<ScalarT,EigenDim> point(data_map_.rows());
            for (size_t i = 0; i < data_map_.rows(); i++) {
                point[i] = grid_point[i]*bin_size_[i];
            }
            return point;
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        Vector<ScalarT,EigenDim> bin_size_;
        Vector<ScalarT,EigenDim> bin_size_inv_;

        GridBinMap grid_lookup_table_;
        std::vector<typename GridBinMap::iterator> bin_iterators_;

        inline void build_index_(const AccumulatorProxy &accum_proxy) {
            if (data_map_.cols() == 0) return;

            bin_iterators_.reserve(data_map_.cols());
            GridPoint grid_coords(data_map_.rows());

            for (size_t i = 0; i < data_map_.cols(); i++) {
                for (size_t j = 0; j < data_map_.rows(); j++) {
                    grid_coords[j] = std::floor(data_map_(j,i)*bin_size_inv_[j]);
                }
                auto lb = grid_lookup_table_.lower_bound(grid_coords);
                if (lb != grid_lookup_table_.end() && !(grid_lookup_table_.key_comp()(grid_coords, lb->first))) {
                    accum_proxy.addToAccumulator(lb->second, i);
                } else {
                    bin_iterators_.emplace_back(grid_lookup_table_.emplace_hint(lb, grid_coords, accum_proxy.buildAccumulator(i)));
                }
            }
        }
    };
}
