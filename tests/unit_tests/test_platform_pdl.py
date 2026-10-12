import pytest
import torch

from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize('major,expected', [(8, False), (9, True), (10, True)])
def test_nvidia_pdl_uses_current_device_capability(monkeypatch, major, expected):
    monkeypatch.setattr(PlatformFL, 'vendor_name', 'nvidia')
    monkeypatch.setattr(PlatformFL, 'device_type', 'cuda')
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    queried = []
    def capability(device):
        queried.append(device)
        return major, 0
    monkeypatch.setattr(torch.cuda, 'get_device_capability', capability)
    assert PlatformFL.is_arch_support_pdl() is expected
    assert queried == [3]


@pytest.mark.parametrize('vendor', ['metax', 'kunlunxin', 'iluvatar'])
def test_cuda_like_vendors_do_not_enable_nvidia_pdl(monkeypatch, vendor):
    monkeypatch.setattr(PlatformFL, 'vendor_name', vendor)
    monkeypatch.setattr(PlatformFL, 'device_type', 'cuda')
    def unexpected_cuda_query():
        raise AssertionError('Non-NVIDIA platform must not query NVIDIA PDL')
    monkeypatch.setattr(torch.cuda, 'current_device', unexpected_cuda_query)
    assert PlatformFL.is_arch_support_pdl() is False


def test_nvidia_pdl_is_false_without_accessible_device(monkeypatch):
    monkeypatch.setattr(PlatformFL, 'vendor_name', 'nvidia')
    monkeypatch.setattr(PlatformFL, 'device_type', 'cuda')
    def unavailable():
        raise RuntimeError('No CUDA device')
    monkeypatch.setattr(torch.cuda, 'current_device', unavailable)
    assert PlatformFL.is_arch_support_pdl() is False
