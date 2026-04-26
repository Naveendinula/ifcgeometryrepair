from __future__ import annotations

import re
from pathlib import Path


STATIC_DIR = Path("app") / "static"


def test_upload_form_uses_inline_parameter_validation() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert re.search(r'<form\b[^>]*id="upload-form"[^>]*\bnovalidate\b', html)
    assert "Î" not in html
    assert "Alpha Wrap &alpha; (m)" in html
    assert "Wrap &alpha; (m)" in html

    numeric_fields = {
        "internal-boundary-threshold-input": {
            "value": "0.30",
            "min": "0",
            "step": "0.01",
            "error_id": "internal-boundary-threshold-error",
        },
        "alpha-wrap-alpha-input": {
            "value": "1.00",
            "min": "0",
            "step": "0.01",
            "error_id": "alpha-wrap-alpha-error",
        },
        "alpha-wrap-offset-input": {
            "value": "0.01",
            "min": "0",
            "step": "0.001",
            "error_id": "alpha-wrap-offset-error",
        },
    }

    for field_id, expected in numeric_fields.items():
        attrs = _input_attrs(html, field_id)
        assert float(attrs["value"]) > 0
        assert attrs["value"] == expected["value"]
        assert attrs["min"] == expected["min"]
        assert attrs["step"] == expected["step"]
        assert attrs["aria-invalid"] == "false"
        assert expected["error_id"] in attrs["aria-describedby"]
        assert f'id="{expected["error_id"]}" class="field-error hidden"' in html

    assert 'min="0.001"' not in html


def test_upload_validation_javascript_is_wired_to_numeric_fields() -> None:
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "const uploadParameterFields = [" in script
    assert "function validateUploadParameters()" in script
    assert "function renderParameterFieldState" in script
    assert "Fix upload parameter values before creating a job." in script

    for form_name in (
        "internal_boundary_thickness_threshold_m",
        "alpha_wrap_alpha_m",
        "alpha_wrap_offset_m",
    ):
        assert form_name in script


def _input_attrs(html: str, element_id: str) -> dict[str, str]:
    match = re.search(rf'<input\b(?=[^>]*id="{re.escape(element_id)}")[^>]*>', html, re.S)
    assert match, f"Could not find input #{element_id}"
    return dict(re.findall(r'([a-zA-Z_:][-a-zA-Z0-9_:]*)="([^"]*)"', match.group(0)))
