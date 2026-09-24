# =============================================================================
#  Single image for the project: it serves as Spark Master, Spark Worker and
#  as the client (driver) that submits the jobs.
#
#  Why ONE image for the 3 roles?
#  In Spark, the Master, the Worker and the Driver run the SAME Spark code:
#  only the Java class started at boot changes. And above all: the driver and
#  the executors MUST have the same Spark and Python version, otherwise Spark
#  refuses to start ("Python in worker has different version").
# =============================================================================

# Official Apache Spark image.
#   3.5.6        -> Spark version
#   scala2.12    -> Spark is written in Scala, compiled here for Scala 2.12
#   java17       -> the JVM in use (Spark runs on the JVM)
#   python3      -> variant shipping Python 3.10 + PySpark
FROM apache/spark:3.5.6-scala2.12-java17-python3-ubuntu

# The image runs by default as the "spark" user (uid 185).
# We switch back to root for the duration of the install.
USER root

# UID/GID of YOUR user on the host (see HOST_UID / HOST_GID in .env).
# Goal: the Parquet files Spark writes into ./data belong to you, instead of
# belonging to an unknown user (otherwise: "Permission denied" when you try to
# delete them from your terminal).
ARG HOST_UID=1000
ARG HOST_GID=1000

RUN groupadd -g "${HOST_GID}" sparkuser 2>/dev/null || true && \
    useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash sparkuser 2>/dev/null || true

# Python dependencies of the application (faker, pandas, pytest...).
# PySpark is not reinstalled: it already comes with the image.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# PostgreSQL JDBC driver: needed to write the Gold tables into the data
# warehouse (phase 13). It goes into /opt/spark/jars so it lands automatically
# on the classpath of the DRIVER **and** of every EXECUTOR.
# (Alternative: --jars at submit time, but Spark then has to copy the jar to
# every executor on every launch.)
ADD https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar \
    /opt/spark/jars/postgresql-42.7.4.jar
RUN chmod 644 /opt/spark/jars/postgresql-42.7.4.jar

# Working directory: the project is mounted here as a volume (see compose).
WORKDIR /opt/workspace

# PYTHONPATH:
#  - /opt/workspace                -> "from src.spark.utils import ..."
#  - /opt/spark/python + py4j      -> makes "import pyspark" work with a plain
#    "python3". spark-submit does this automatically, but pytest and scripts
#    launched by hand do not (phase 15).
ENV PYTHONPATH=/opt/workspace:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
# PYSPARK_PYTHON: which Python interpreter the executors must use.
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
# The spark-submit / spark-shell / pyspark binaries directly on the PATH.
ENV PATH="${PATH}:/opt/spark/bin:/opt/spark/sbin"
# Explicit HOME: Spark and pip write caches there (.ivy2, .cache).
ENV HOME=/home/sparkuser

# Folders Spark writes to (master/worker logs, temporary shuffle files).
RUN mkdir -p /opt/spark/logs /opt/spark/work /tmp/spark-events && \
    chown -R "${HOST_UID}:${HOST_GID}" /opt/spark/logs /opt/spark/work /tmp/spark-events /opt/workspace /home/sparkuser

USER sparkuser
