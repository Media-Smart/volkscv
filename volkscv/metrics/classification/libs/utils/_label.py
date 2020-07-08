# modify from sklearn (0.22.1)
import numpy as np
import scipy.sparse as sp

from ._base import check_array
from .multiclass import type_of_target, unique_labels
from .validation import column_or_1d, _num_samples, check_is_fitted
from .base import BaseEstimator, TransformerMixin


def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape [n_classes]
        Uniquely holds the label for each class.

    neg_label : int (default: 0)
        Value with which negative labels must be encoded.

    pos_label : int (default: 1)
        Value with which positive labels must be encoded.

    sparse_output : boolean (default: False),
        Set to true if output binary array is desired in CSR sparse format

    Returns
    -------
    Y : numpy array or CSR matrix of shape [n_samples, n_classes]
        Shape will be [n_samples, 1] for binary problems.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    See also
    --------
    LabelBinarizer : class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation
    """
    if not isinstance(y, list):
        # XXX Workaround that will be removed when list of list format is
        # dropped
        y = check_array(y, accept_sparse='csr', ensure_2d=False, dtype=None)
    else:
        if _num_samples(y) == 0:
            raise ValueError('y has 0 samples: %r' % y)
    if neg_label >= pos_label:
        raise ValueError("neg_label={0} must be strictly less than "
                         "pos_label={1}.".format(neg_label, pos_label))

    if (sparse_output and (pos_label == 0 or neg_label != 0)):
        raise ValueError("Sparse binarization is only supported with non "
                         "zero pos_label and zero neg_label, got "
                         "pos_label={0} and neg_label={1}"
                         "".format(pos_label, neg_label))

    # To account for pos_label == 0 in the dense case
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label

    y_type = type_of_target(y)
    if 'multioutput' in y_type:
        raise ValueError("Multioutput target data is not supported with label "
                         "binarization")
    if y_type == 'unknown':
        raise ValueError("The type of target data is not known")

    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)

    if y_type == "binary":
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            else:
                Y = np.zeros((len(y), 1), dtype=np.int)
                Y += neg_label
                return Y
        elif len(classes) >= 3:
            y_type = "multiclass"

    sorted_class = np.sort(classes)
    if y_type == "multilabel-indicator":
        y_n_classes = y.shape[1] if hasattr(y, 'shape') else len(y[0])
        if classes.size != y_n_classes:
            raise ValueError("classes {0} mismatch with the labels {1}"
                             " found in the data"
                             .format(classes, unique_labels(y)))

    if y_type in ("binary", "multiclass"):
        y = column_or_1d(y)

        # pick out the known labels from y
        y_in_classes = np.in1d(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = sp.csr_matrix((data, indices, indptr),
                          shape=(n_samples, n_classes))
    elif y_type == "multilabel-indicator":
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError("%s target data is not supported with label "
                         "binarization" % y_type)

    if not sparse_output:
        Y = Y.toarray()
        Y = Y.astype(int, copy=False)

        if neg_label != 0:
            Y[Y == 0] = neg_label

        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        Y.data = Y.data.astype(int, copy=False)

    # preserve label ordering
    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]

    if y_type == "binary":
        if sparse_output:
            Y = Y.getcol(-1)
        else:
            Y = Y[:, -1].reshape((-1, 1))

    return Y


def _encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques


def _encode_numpy(values, uniques=None, encode=False, check_unknown=True):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        if encode:
            uniques, encoded = np.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            # unique sorts
            return np.unique(values)
    if encode:
        if check_unknown:
            diff = _encode_check_unknown(values, uniques)
            if diff:
                raise ValueError("y contains previously unseen labels: %s"
                                 % str(diff))
        encoded = np.searchsorted(uniques, values)
        return uniques, encoded
    else:
        return uniques


def _encode(values, uniques=None, encode=False, check_unknown=True):
    """Helper function to factorize (find uniques) and encode values.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.

    Parameters
    ----------
    values : array
        Values to factorize or encode.
    uniques : array, optional
        If passed, uniques are not determined from passed values (this
        can be because the user specified categories, or because they
        already have been determined in fit).
    encode : bool, default False
        If True, also encode the values into integer codes based on `uniques`.
    check_unknown : bool, default True
        If True, check for values in ``values`` that are not in ``unique``
        and raise an error. This is ignored for object dtype, and treated as
        True in this case. This parameter is useful for
        _BaseEncoder._transform() to avoid calling _encode_check_unknown()
        twice.

    Returns
    -------
    uniques
        If ``encode=False``. The unique values are sorted if the `uniques`
        parameter was None (and thus inferred from the data).
    (uniques, encoded)
        If ``encode=True``.

    """
    if values.dtype == object:
        try:
            res = _encode_python(values, uniques, encode)
        except TypeError:
            raise TypeError("argument must be a string or number")
        return res
    else:
        return _encode_numpy(values, uniques, encode,
                             check_unknown=check_unknown)


def _encode_check_unknown(values, uniques, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : array
        Values to check for unknowns.
    uniques : array
        Allowed uniques values.
    return_mask : bool, default False
        If True, return a mask of the same shape as `values` indicating
        the valid values.

    Returns
    -------
    diff : list
        The unique values present in `values` and not in `uniques` (the
        unknown values).
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.

    """
    if values.dtype == object:
        uniques_set = set(uniques)
        diff = list(set(values) - uniques_set)
        if return_mask:
            if diff:
                valid_mask = np.array([val in uniques_set for val in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff
    else:
        unique_values = np.unique(values)
        diff = list(np.setdiff1d(unique_values, uniques, assume_unique=True))
        if return_mask:
            if diff:
                valid_mask = np.in1d(values, uniques)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff


class LabelEncoder(TransformerMixin, BaseEstimator):
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    .. versionadded:: 0.12

    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    See also
    --------
    sklearn.preprocessing.OrdinalEncoder : Encode categorical features
        using an ordinal encoding scheme.

    sklearn.preprocessing.OneHotEncoder : Encode categorical features
        as a one-hot numeric array.
    """

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _encode(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _encode(y, encode=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        _, y = _encode(y, uniques=self.classes_, encode=True)
        return y

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if len(diff):
            raise ValueError(
                    "y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]

    def _more_tags(self):
        return {'X_types': ['1dlabels']}

