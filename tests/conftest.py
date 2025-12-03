import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "music.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
        CREATE TABLE Album (
            AlbumId INTEGER PRIMARY KEY, Title TEXT NOT NULL,
            ArtistId INTEGER NOT NULL REFERENCES Artist(ArtistId)
        );
        INSERT INTO Artist VALUES (1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith');
        INSERT INTO Album VALUES (1, 'For Those About To Rock', 1), (2, 'Let There Be Rock', 1),
                                 (3, 'Balls to the Wall', 2);
        """
    )
    conn.commit()
    conn.close()
    return path

