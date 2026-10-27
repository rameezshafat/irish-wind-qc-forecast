import io

from fastapi.testclient import TestClient

from windqc.api import app
from windqc.data import load_met_hourly

SAMPLE = """Station Name: TEST
Station Height: 71 M

date:  -  Date and Time (utc)

date,ind,rain,ind,temp,ind,wetb,dewpt,vappr,rhum,msl,ind,wdsp,ind,wddir,ww,w,sun,vis,clht,clamt
01-jan-2024 00:00,0,0.0,0,7.8,0,5.8,3.3,7.6,72,1015.3,2,19,2,160,2,6,0.0,10000,19,8
01-jan-2024 01:00,0,0.6,0,7.2,0,6.0,4.4,8.4,83,1015.2,4,20,2,160,21,6,0.0,20000,17,8
01-jan-2024 02:00,0,0.1,0,7.2,0,6.0,4.4,8.3,82,1015.2,2, ,2,160,21,6,0.0,20000,19,8
"""


def test_load_met_hourly_parses_header_and_flags():
    df = load_met_hourly(io.StringIO(SAMPLE))
    assert len(df) == 3
    assert str(df.index.tz) == "UTC"
    assert df["wdsp"].tolist()[:2] == [19, 20]
    assert df["wdsp"].isna().iloc[2]
    assert df["wind_defective"].tolist() == [False, True, False]


def test_api_health_and_qc():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    readings = [{"timestamp": f"2024-01-01T{h:02d}:00:00Z", "value": 7.0} for h in range(8)]
    out = client.post("/qc", json={"readings": readings}).json()
    assert len(out) == 8
    assert all(row["flatline"] for row in out)
