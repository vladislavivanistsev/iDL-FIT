/* global expect */

describe('Calculations tests', function () {

    it('calculateSurfaceCharges must return correct value.', function () {
        expect(calculateSurfaceCharges(0.362, 0.667, 1.000, {"d": 0.12}, 0.20, [-2, -1.96])).toEqual([-2.535141644212017, -2.4849258559180356]);
    });
});