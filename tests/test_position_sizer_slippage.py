import pytest
import math
from core.trading.position_sizer import (
    VolatilityAwarePositionSizer,
    AC_ETA,
    AC_GAMMA,
    ADV_ILLIQUID_THRESHOLD,
    ADV_SEMILIQUID_THRESHOLD,
    SPREAD_BPS_ILLIQUID,
    SPREAD_BPS_SEMILIQUID,
    SPREAD_BPS_LIQUID
)

def test_slippage_fallback_invalid_adv():
    """Test fallback when adv_20d is <= 0"""
    # Should return SPREAD_BPS_SEMILIQUID
    assert VolatilityAwarePositionSizer.almgren_chriss_slippage(1000000, 0) == SPREAD_BPS_SEMILIQUID
    assert VolatilityAwarePositionSizer.almgren_chriss_slippage(1000000, -100) == SPREAD_BPS_SEMILIQUID

def test_slippage_liquidity_tiers():
    """Test half-spread cost for different liquidity tiers"""
    # Small participation to minimize impact
    trade_val = 1.0

    # Illiquid: < 5e7 (e.g., 4e7)
    adv_illiquid = ADV_ILLIQUID_THRESHOLD - 1000
    res_illiquid = VolatilityAwarePositionSizer.almgren_chriss_slippage(trade_val, adv_illiquid)
    expected_illiquid = (AC_ETA * (trade_val/adv_illiquid)**0.5 +
                         AC_GAMMA * (trade_val/adv_illiquid) +
                         SPREAD_BPS_ILLIQUID)
    assert pytest.approx(res_illiquid) == expected_illiquid

    # Semi-liquid: 5e7 <= adv < 5e8 (e.g., 1e8)
    adv_semi = 1e8
    res_semi = VolatilityAwarePositionSizer.almgren_chriss_slippage(trade_val, adv_semi)
    expected_semi = (AC_ETA * (trade_val/adv_semi)**0.5 +
                     AC_GAMMA * (trade_val/adv_semi) +
                     SPREAD_BPS_SEMILIQUID)
    assert pytest.approx(res_semi) == expected_semi

    # Liquid: >= 5e8 (e.g., 1e9)
    adv_liquid = ADV_SEMILIQUID_THRESHOLD + 1000
    res_liquid = VolatilityAwarePositionSizer.almgren_chriss_slippage(trade_val, adv_liquid)
    expected_liquid = (AC_ETA * (trade_val/adv_liquid)**0.5 +
                       AC_GAMMA * (trade_val/adv_liquid) +
                       SPREAD_BPS_LIQUID)
    assert pytest.approx(res_liquid) == expected_liquid

def test_slippage_impact_calculations():
    """Test temporary and permanent impact calculations"""
    adv = 1e9 # Liquid tier (>= 5e8)
    trade_val = 1e7 # 1% participation

    participation_rate = trade_val / adv
    temp_impact = AC_ETA * (participation_rate ** 0.5)
    perm_impact = AC_GAMMA * participation_rate
    spread_cost = SPREAD_BPS_LIQUID

    expected = temp_impact + perm_impact + spread_cost
    result = VolatilityAwarePositionSizer.almgren_chriss_slippage(trade_val, adv)

    assert pytest.approx(result) == expected

    # Manually calculate for verification:
    # participation = 0.01
    # temp_impact = 0.0025 * sqrt(0.01) = 0.0025 * 0.1 = 0.00025
    # perm_impact = 0.001 * 0.01 = 0.00001
    # spread_cost = 0.0003
    # total = 0.00025 + 0.00001 + 0.0003 = 0.00056
    assert pytest.approx(result) == 0.00056

def test_slippage_scaling():
    """Test that slippage increases with trade value"""
    adv = 1e9
    low_val = 1e6
    high_val = 1e7

    slip_low = VolatilityAwarePositionSizer.almgren_chriss_slippage(low_val, adv)
    slip_high = VolatilityAwarePositionSizer.almgren_chriss_slippage(high_val, adv)

    assert slip_high > slip_low
