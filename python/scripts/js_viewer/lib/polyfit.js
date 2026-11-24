// Source: https://github.com/rfink/polyfit.js
// See also: https://arachnoid.com/polysolve/
Polyfit = (function () {
    /**
     * Polyfit
     * @constructor
     * @param {number[]|Float32Array|Float64Array} x
     * @param {number[]|Float32Array|Float64Array} y
     */
    function Polyfit(x, y) {
        // Make sure we return an instance
        if (!(this instanceof Polyfit)) {
            return new Polyfit(x, y);
        }
        // Check that x any y are both arrays of the same type
        if (!((x instanceof Array && y instanceof Array) ||
            (x instanceof Float32Array && y instanceof Float32Array) ||
            (x instanceof Float64Array && y instanceof Float64Array))) {
            throw new Error('x and y must be arrays');
        }
        if (x instanceof Float32Array) {
            this.FloatXArray = Float32Array;
        }
        else if (x instanceof Float64Array) {
            this.FloatXArray = Float64Array;
        }
        // Make sure we have equal lengths
        if (x.length !== y.length) {
            throw new Error('x and y must have the same length');
        }
        this.x = x;
        this.y = y;
    }
    /**
     * Perform gauss-jordan division
     *
     * @param {number[][]|Float32Array[]|Float64Array[]} matrix - gets modified
     * @param {number} row
     * @param {number} col
     * @param {number} numCols
     * @returns void
     */
    Polyfit.gaussJordanDivide = function (matrix, row, col, numCols) {
        for (var i = col + 1; i < numCols; i++) {
            matrix[row][i] /= matrix[row][col];
        }
        matrix[row][col] = 1;
    };
    /**
     * Perform gauss-jordan elimination
     *
     * @param {number[][]|Float64Array[]} matrix - gets modified
     * @param {number} row
     * @param {number} col
     * @param {number} numRows
     * @param {number} numCols
     * @returns void
     */
    Polyfit.gaussJordanEliminate = function (matrix, row, col, numRows, numCols) {
        for (var i = 0; i < numRows; i++) {
            if (i !== row && matrix[i][col] !== 0) {
                for (var j = col + 1; j < numCols; j++) {
                    matrix[i][j] -= matrix[i][col] * matrix[row][j];
                }
                matrix[i][col] = 0;
            }
        }
    };
    /**
     * Perform gauss-jordan echelon method
     *
     * @param {number[][]|Float32Array[]|Float64Array[]} matrix - gets modified
     * @returns {number[][]|Float32Array[]|Float64Array[]} matrix
     */
    Polyfit.gaussJordanEchelonize = function (matrix) {
        var rows = matrix.length;
        var cols = matrix[0].length;
        var i = 0;
        var j = 0;
        var k;
        var swap;
        while (i < rows && j < cols) {
            k = i;
            // Look for non-zero entries in col j at or below row i
            while (k < rows && matrix[k][j] === 0) {
                k++;
            }
            // If an entry is found at row k
            if (k < rows) {
                // If k is not i, then swap row i with row k
                if (k !== i) {
                    swap = matrix[i];
                    matrix[i] = matrix[k];
                    matrix[k] = swap;
                }
                // If matrix[i][j] is != 1, divide row i by matrix[i][j]
                if (matrix[i][j] !== 1) {
                    Polyfit.gaussJordanDivide(matrix, i, j, cols);
                }
                // Eliminate all other non-zero entries
                Polyfit.gaussJordanEliminate(matrix, i, j, rows, cols);
                i++;
            }
            j++;
        }
        return matrix;
    };
    /**
     * Perform regression
     *
     * @param {number} x
     * @param {number[]|Float32Array[]|Float64Array[]} terms
     * @returns {number}
     */
    Polyfit.regress = function (x, terms) {
        var a = 0;
        var exp = 0;
        for (var i = 0, len = terms.length; i < len; i++) {
            a += terms[i] * Math.pow(x, exp++);
        }
        return a;
    };
    /**
     * Compute correlation coefficient
     *
     * @param {number[]|Float32Array[]|Float64Array[]} terms
     * @returns {number}
     */
    Polyfit.prototype.correlationCoefficient = function (terms) {
        var r = 0;
        var n = this.x.length;
        var sx = 0;
        var sx2 = 0;
        var sy = 0;
        var sy2 = 0;
        var sxy = 0;
        var x;
        var y;
        for (var i = 0; i < n; i++) {
            x = Polyfit.regress(this.x[i], terms);
            y = this.y[i];
            sx += x;
            sy += y;
            sxy += x * y;
            sx2 += x * x;
            sy2 += y * y;
        }
        var div = Math.sqrt((sx2 - (sx * sx) / n) * (sy2 - (sy * sy) / n));
        if (div !== 0) {
            r = Math.pow((sxy - (sx * sy) / n) / div, 2);
        }
        return r;
    };
    /**
     * Run standard error function
     *
     * @param {number[]|Float32Array[]|Float64Array[]} terms
     * @returns number
     */
    Polyfit.prototype.standardError = function (terms) {
        var r = 0;
        var n = this.x.length;
        if (n > 2) {
            var a = 0;
            for (var i = 0; i < n; i++) {
                a += Math.pow((Polyfit.regress(this.x[i], terms) - this.y[i]), 2);
            }
            r = Math.sqrt(a / (n - 2));
        }
        return r;
    };
    /**
     * Compute coefficients for given data matrix
     *
     * @param {number} p
     * @returns {number[]|Float32Array|Float64Array}
     */
    Polyfit.prototype.computeCoefficients = function (p) {
        var n = this.x.length;
        var r;
        var c;
        var rs = 2 * (++p) - 1;
        var i;
        var m = [];
        // Initialize array with 0 values
        if (this.FloatXArray) {
            // fast FloatXArray-Matrix init
            var bytesPerRow = (p + 1) * this.FloatXArray.BYTES_PER_ELEMENT;
            var buffer = new ArrayBuffer(p * bytesPerRow);
            for (i = 0; i < p; i++) {
                m[i] = new this.FloatXArray(buffer, i * bytesPerRow, p + 1);
            }
        }
        else {
            var zeroRow = [];
            for (i = 0; i <= p; i++) {
                zeroRow[i] = 0;
            }
            m[0] = zeroRow;
            for (i = 1; i < p; i++) {
                // copy zeroRow
                m[i] = zeroRow.slice();
            }
        }
        var mpc = [n];
        for (i = 1; i < rs; i++) {
            mpc[i] = 0;
        }
        for (i = 0; i < n; i++) {
            var x = this.x[i];
            var y = this.y[i];
            // Process precalculation array
            for (r = 1; r < rs; r++) {
                mpc[r] += Math.pow(x, r);
            }
            // Process RH column cells
            m[0][p] += y;
            for (r = 1; r < p; r++) {
                m[r][p] += Math.pow(x, r) * y;
            }
        }
        // Populate square matrix section
        for (r = 0; r < p; r++) {
            for (c = 0; c < p; c++) {
                m[r][c] = mpc[r + c];
            }
        }
        Polyfit.gaussJordanEchelonize(m);
        var terms = this.FloatXArray && new this.FloatXArray(m.length) || [];
        for (i = m.length - 1; i >= 0; i--) {
            terms[i] = m[i][p];
        }
        return terms;
    };
    /**
     * Using given degree of fitment, return a function that will calculate
     * the y for a given x
     *
     * @param {number} degree  > 0
     * @returns {Function}     f(x) =
     */
    Polyfit.prototype.getPolynomial = function (degree) {
        if (isNaN(degree) || degree < 0) {
            throw new Error('Degree must be a positive integer');
        }
        var terms = this.computeCoefficients(degree);
        var eqParts = [];
        eqParts.push(terms[0].toPrecision());
        for (var i = 1, len = terms.length; i < len; i++) {
            eqParts.push(terms[i] + ' * Math.pow(x, ' + i + ')');
        }
        var expr = 'return ' + eqParts.join(' + ') + ';';
        /* jshint evil: true */
        return new Function('x', expr);
        /* jshint evil: false */
    };
    /**
     * Convert the polynomial to a string expression, mostly useful for visual
     * debugging
     *
     * @param {number} degree
     * @returns {string}
     */
    Polyfit.prototype.toExpression = function (degree) {
        if (isNaN(degree) || degree < 0) {
            throw new Error('Degree must be a positive integer');
        }
        var terms = this.computeCoefficients(degree);
        var eqParts = [];
        var len = terms.length;
        eqParts.push(terms[0].toPrecision());
        for (var i = 1; i < len; i++) {
            eqParts.push(terms[i] + 'x^' + i);
        }
        return eqParts.join(' + ');
    };
    return Polyfit;
})();