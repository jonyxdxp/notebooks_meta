



// from https://github.com/dilawar/moden-hopfield-network/blob/main/src/memory.rs




















//! Memory.

use num_traits::Float;

pub const NUM_SUBCATEGORY: usize = 4;

/// Memory stored in DAM.
///
/// A pattern's `data` is a vector that can be decomposed into 3 subvectors (F, C1, C2).
///
/// F:  Feature vector of size `feature_size`. Supplied by user.
/// C1: Class vector whose size is equal to number of possible labels or classes. For MNIST digits,
///     size of C1 would be 10. If F belongs to class i then C1[i] = 1.0 and C1[j/=i] = -1.0.
/// C2: Subclass vector (for internal use). Its size is the constant NUM_SUBCATEGORY.
#[derive(Debug, PartialEq, Eq)]
pub struct Memory<T> {
    /// pattern as well as the appended classification padding.
    /// total size of the data is length of the pattern + number of categories.
    data: Vec<T>,

    /// size of the feature.
    feature_size: usize,

    /// Category of the pattern after classification.
    category: Option<usize>,

    /// subcategory of the pattern. Used internally.
    subcategory: Option<usize>,
}

impl std::fmt::Display for Memory<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr_polar_binary())
    }
}

impl<T: Float + Default + std::iter::Sum + std::fmt::Debug> Memory<T> {
    /// create a new [Memory] without category and subcategory.
    pub fn new_without_category(pattern: &[T]) -> Self {
        Memory {
            data: pattern.to_vec(),
            feature_size: pattern.len(),
            category: None,
            subcategory: None,
        }
    }

    /// Create a new [Memory]
    pub fn new(
        pattern: &[T],
        category: Option<usize>,
        subcategory: Option<usize>,
        max_categories: usize,
    ) -> Self {
        let pattern_len = pattern.len();
        // `NUM_SUBCATEGORY` is the number of sub-categories each category can have.
        let mut data = Vec::with_capacity(pattern_len + max_categories + NUM_SUBCATEGORY);
        data.extend_from_slice(pattern);

        // Initialize the vector with -1.0.
        data.resize(pattern_len + max_categories + NUM_SUBCATEGORY, -T::one());

        // set the category.
        if let Some(cat) = category {
            data[pattern_len + cat] = T::one();
        }

        if let Some(subcat) = subcategory {
            data[pattern_len + max_categories + subcat] = T::one();
        }

        // return the pattern.
        Memory {
            data,
            feature_size: pattern_len,
            category,
            subcategory,
        }
    }

    /// Generate a random pattern of given size.
    pub fn random(size: usize, size_of_classes: usize) -> Self {
        let img: Vec<T> = (0..size)
            .map(|_| {
                if rand::random::<f32>() > 0.5 {
                    T::one()
                } else {
                    -T::one()
                }
            })
            .collect();
        Memory::new(
            &img,
            Some(rand::random::<usize>() % size_of_classes),
            None,
            size_of_classes,
        )
    }

    /// returns the number of categories this pattern can belong to.
    pub fn cardinality_of_class(&self) -> usize {
        self.data.len() - self.feature_size - NUM_SUBCATEGORY
    }

    pub fn from_pattern(pat: &Self) -> Self {
        // return the pattern.
        Memory {
            data: pat.data.to_vec(),
            feature_size: pat.feature_size,
            category: pat.category,
            subcategory: pat.subcategory,
        }
    }

    /// category of the pattern.
    pub fn get_category(&self) -> Option<usize> {
        if let Some(cat) = self.category {
            assert_eq!(self.data[self.feature_size + cat], T::one());
        }
        self.category
    }

    /// Is this pattern has category `cat`
    pub fn is_category(&self, cat: usize) -> bool {
        assert!(self.category.is_some(), "Memory is not classified yet.");
        self.category.unwrap() == cat
    }

    /// Size of the pattern. Size of data is larger than pattern.
    pub fn get_feature_size(&self) -> usize {
        self.feature_size
    }

    /// Full size of pattern. Memory + classification.
    #[inline(always)]
    pub fn get_data_size(&self) -> usize {
        self.data.len()
    }

    /// Get the state of the DAM.
    #[inline(always)]
    pub fn get_state(&self, i: usize) -> T {
        self.data[i]
    }

    /// alias for data (full vector).
    #[inline(always)]
    pub fn sigma(&self) -> &[T] {
        &self.data
    }

    /// Set the sigma (state).
    #[inline(always)]
    pub fn set_sigma(&mut self, vec: &[T]) {
        self.data[0..vec.len()].copy_from_slice(vec);
    }

    /// alias for data (full vector).
    #[inline(always)]
    pub fn get_feature_vec(&self) -> &[T] {
        &self.data[0..self.feature_size]
    }

    /// class vector.
    #[inline(always)]
    pub fn get_class_vec(&self) -> &[T] {
        &self.data[self.feature_size..self.data.len() - NUM_SUBCATEGORY]
    }

    /// Get the class of the [Pattern].
    #[inline(always)]
    pub fn get_class(&self) -> Option<usize> {
        self.get_class_vec().iter().position(|&x| x == T::one())
    }

    /// Get the subclass as slice.
    #[inline(always)]
    pub fn get_subclass_vec(&self) -> &[T] {
        &self.data[self.data.len() - NUM_SUBCATEGORY..]
    }

    /// Get the subclass index.
    pub fn get_subclass(&self) -> Option<usize> {
        self.get_subclass_vec().iter().position(|&x| x == T::one())
    }

    // /// Clamp the data to state.
    // fn clamp_input(&mut self, input: &[T]) {
    //     assert!(
    //         input.len() == self.feature_size,
    //         "Input size is larger than pattern size"
    //     );
    //     self.data[..input.len()].copy_from_slice(input);
    // }

    /// norm squared.
    pub fn norm_squared(&self) -> T {
        self.get_feature_vec().iter().map(|x| x.powi(2)).sum()
    }

    /// Polar binary format.
    pub fn repr_polar_binary(&self) -> String {
        format!(
            "{:?} {}",
            self.category,
            crate::helper::repr_polar_binary(&self.data),
        )
    }

    /// Add noise to pattern.
    pub fn add_noise(&mut self, v: f32) {
        crate::numeric::add_noise_mut(&mut self.data[0..self.feature_size], v);
    }

    /// Polar binary format.
    pub fn repr_polar_image(&self, cols: Option<usize>) -> String {
        let num_cols = cols.unwrap_or((self.feature_size as f64).sqrt() as usize);
        crate::helper::repr_polar_image(&self.data, num_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_sanity() {
        let p1 = Memory::new(&[0.0, 1.0, 1.0], Some(1), None, 2);
        assert_eq!(p1.get_feature_vec(), vec![0.0, 1.0, 1.0]);
        assert_eq!(p1.get_class_vec(), vec![-1.0, 1.0]);
        assert_eq!(p1.get_class(), Some(1));
        assert_eq!(p1.get_subclass_vec(), vec![-1.0; NUM_SUBCATEGORY]);
        assert_eq!(p1.get_subclass(), None);

        let p2 = Memory::new(&[0.0, 1.0, 1.0], Some(0), None, 2);
        assert_eq!(p2.get_feature_vec(), vec![0.0, 1.0, 1.0]);
        assert_eq!(p2.get_class_vec(), vec![1.0, -1.0]);
        assert_eq!(p2.get_class(), Some(0));
        assert_eq!(p2.get_subclass_vec(), vec![-1.0; NUM_SUBCATEGORY]);
        assert_eq!(p2.get_subclass(), None);
    }

    #[test]
    fn test_pattern_simple() {
        let psize = 100;
        let img: Vec<f32> = (0..psize)
            .map(|_| {
                if rand::random::<f32>() > 0.5 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let pattern = Memory::new(&img, Some(1), None, 10);
        println!("{}", pattern.repr_polar_binary());
        assert_eq!(pattern.get_feature_size(), psize);
        assert_eq!(pattern.get_data_size(), psize + 10 + NUM_SUBCATEGORY);
        assert_eq!(pattern.get_class(), Some(1));
    }
}