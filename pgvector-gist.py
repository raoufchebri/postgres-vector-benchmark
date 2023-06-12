import h5py
import psycopg2
import time
import numpy

# For best performance:
# - Use socket connection
# - Use prepared statements (todo)
# - Use binary format (todo)
# - Increase work_mem? (todo)
# - Increase shared_buffers? (todo)
# - Try different drivers (todo)
class Pgvector:
    def __init__(self, metric, lists):
        self._metric = metric

        self._lists = lists
        self._probes = None

        self._opclass = {'angular': 'vector_cosine_ops', 'euclidean': 'vector_l2_ops'}[metric]
        self._op = {'angular': '<=>', 'euclidean': '<->'}[metric]
        self._table = 'vectors_%s_%d' % (metric, lists)
        self._query = "SELECT id FROM %s ORDER BY vec %s %%s::vector LIMIT %%s" % (self._table, self._op)

        self._conn = psycopg2.connect('postgresql://raouf@ep-lucky-bush-021504.eu-west-1.aws.neon.build/neondb')
        self._conn.autocommit = True
        self._cur = self._conn.cursor()
        self._cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    def fit(self, X):
        self._cur.execute('CREATE TABLE IF NOT EXISTS %s (id integer, vec vector(%d))' % (self._table, X.shape[1]))

        f = open('gist.csv','w+')
        for i, x in enumerate(X):
            f.write(str(i))
            f.write('\t')
            f.write('[' + ','.join(str(v) for v in x.tolist()) + ']')
            f.write('\n')
        f.seek(0)
        self._cur.copy_from(f, self._table, columns=('id', 'vec'))

        self._cur.execute('CREATE INDEX ON %s USING ivfflat (vec %s) WITH (lists = %d)' % (self._table, self._opclass, self._lists))
        self._cur.execute('ANALYZE %s' % self._table)

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute('SET ivfflat.probes = %s', (str(probes),))
        self._cur.execute('SET effective_io_concurrency=10')
        self._cur.execute('SET enable_seqscan=off')

    def query(self, v, n):
        self._cur.execute(self._query, (v.tolist(), n))
        res = self._cur.fetchall()
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)


def get_dataset(hdf5_fn):
    hdf5_f = h5py.File(hdf5_fn, 'r')
    dimension = int(hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])
    return hdf5_f, dimension

def main():
    lists = 100
    probes = 9
    metric = 'euclidean'
    knn = 10 #  Number of nearest neighbours for the algorithm to return.',
    algo = Pgvector(metric, lists)
    dataset = 'gist-960-euclidean.hdf5'
    D, dimension = get_dataset(dataset)
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])
    distance = D.attrs['distance']
    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('got %d queries' % len(X_test))

    # start_time = time.time()
    # algo.fit(X_train)
    build_time = time.time()
    # print('Built index in', build_time - start_time)

    algo.set_query_arguments(probes)
    for x in X_test:
        algo.query(x, knn)
        break

    query_time =  time.time() - build_time
    print('Run queries in', query_time)


main()

