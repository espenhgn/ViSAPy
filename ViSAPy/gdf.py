"""
Create db from gdf files to be able to then select neuron spike times.

Best use case:
1. run simulation
2. create sqlite db of spike times with indexing
3. use this db many times

Creating index for db will dominate insertions for larger set of spike times.

[TODO] Check how much slower things are if index is created at start
[TODO] Simplify block_read. sqlite probably can covert for insert.
[TODO] Error checking
[TODO] sqlite optimizations
[TODO] Create read buffers once instead of for each file
[TODO] Spike times may be storable as int32 rather than float, save space
"""
import numpy as np
import sqlite3 as sqlite
import os, glob
from time import time as now
import matplotlib.pyplot as plt


plt.rcdefaults()
plt.rcParams.update({
    'font.size' : 16,
    'axes.labelsize' : 16,
    'axes.titlesize' : 16,
    'legend.fontsize' : 14,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16,
    'figure.subplot.wspace' : 0.3,
    'figure.subplot.hspace' : 0.3,
})

class GDF(object):
    """
    1. Read from gdf files
    2. Create sqlite db of (neuron, spike time)
    3. Query spike times for neurons
    """

    def __init__(self, dbname, bsize=int(1e6), new_db=True,
                 debug=False):
        """

        Parameters:
            ::

                dbname : str,
                    filename of sqlite database
                bsize : int,
                    number of spike times to insert
                new_db : bool,
                    new database with name dbname, will overwrite
                    at a time, determines memory usage
        """
        if new_db:
            try:
                os.unlink(dbname)
            except:
                print 'creating new database file %s' % dbname

        self.conn = sqlite.connect(dbname)
        self.cursor = self.conn.cursor()
        self.bsize = bsize
        self.debug = debug

    def _blockread(self, fname):
        """
        Generator yields bsize lines from gdf file

        Parameters:
            ::

                fname : str,
                    name of gdf-file
        """
        with open(fname, 'rb') as f:
            while True:
                a = []
                for i in range(self.bsize):
                    line = f.readline()
                    if not line: break
                    a.append(line.split())
                if a == []: raise StopIteration
                yield a
        f.close()
   
    def create(self, re='brunel-py-ex-*.gdf', index=True):
        """
        Create db from list of gdf file glob

        Parameters:
            ::

                re : str,
                    file glob to load
                index : bool,
                    create index on neurons for speed
        """
        self.cursor.execute('CREATE TABLE IF NOT EXISTS spikes (neuron INT UNSIGNED, time REAL)')

        tic = now()
        for f in glob.glob(re):
            print f
            while True:
                try:
                    for data in self._blockread(f):
                        self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
                        self.conn.commit()
                except:
                    continue
                break                
        toc = now()
        if self.debug: print 'Inserts took %g seconds.' % (toc-tic)

        # optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print 'Indexed db in %g seconds.' % (toc-tic)

    def create_from_list(self, re=[], index=True):
        '''
        create db from list of arrays

        Parameters:
            ::

                re : list,
                    index of element is cell index, and element i an array of spike times in ms
                index : bool,
                    create index on neurons for speed
        '''
        self.cursor.execute('CREATE TABLE IF NOT EXISTS spikes (neuron INT UNSIGNED, time REAL)')

        tic = now()
        i = 0
        for x in re:
            data = zip([i] * len(x), x)
            self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
            i += 1
        self.conn.commit()
        toc = now()
        if self.debug: print 'Inserts took %g seconds.' % (toc-tic)

        # optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print 'Indexed db in %g seconds.' % (toc-tic)


    def select(self, neurons):
        """
        Select spike trains.

        Parameters:
            ::

                neurons : np.array or list of neurons

        Returns:
            ::

                s : list of np.array's
                    spike times


        """
        s = []
        for neuron in neurons:
            self.cursor.execute('SELECT time FROM spikes where neuron = %d' % neuron)
            sel = self.cursor.fetchall()
            spikes = np.array(sel).flatten()
            s.append(spikes)
        return s

    def interval(self, T=[0, 1000]):
        """
        Get all spikes in a time interval T

        Parameters:
            ::

                T : list,
                    time interval

        Returns:
            ::

                s : list,
                    nested list with spike times
        """
        self.cursor.execute('SELECT * FROM spikes WHERE time BETWEEN %f AND %f' % tuple(T))
        sel = self.cursor.fetchall()
        return sel

    def select_neurons_interval(self, neurons, T=[0, 1000]):
        """
        Get all spikes from neurons in a time interval T.

        Parameters:
            ::

                T : list,
                    time interval

        Returns:
            ::

                s : list,
                    nested list with spike times

        """
        s = []
        for neuron in neurons:
            self.cursor.execute('SELECT time FROM spikes WHERE time BETWEEN %f AND %f and neuron = %d'  % (T[0], T[1], neuron))
            sel = self.cursor.fetchall()
            spikes = np.array(sel).flatten()

            s.append(spikes)

        return s

    def neurons(self):
        """
        Return list of neuron indices

        Returns:
            ::

                list

        """
        self.cursor.execute('SELECT DISTINCT neuron FROM spikes ORDER BY neuron')
        sel = self.cursor.fetchall()
        return np.array(sel).flatten()

    def num_spikes(self):
        """
        Return total number of spikes

        Returns:
            ::

                list


        """
        self.cursor.execute('SELECT Count(*) from spikes')
        rows = self.cursor.fetchall()[0]
        # Check against 'wc -l *ex*.gdf'
        if self.debug: print 'DB has %d spikes' % rows
        return rows

    def close(self):
        """


        """
        self.cursor.close()
        self.conn.close()

    def plotstuff(self, T=[0, 1000]):
        '''
        create a scatter plot of the contents of the database,
        with entries on the interval T

        Parameters:
            ::

                T : list,
                    time interval


        '''

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

        neurons = self.neurons()
        i = 0
        for x in self.select_neurons_interval(neurons, T):
            ax.plot(x, np.zeros(x.size) + neurons[i], 'o',
                    markersize=1, markerfacecolor='k', markeredgecolor='k',
                    alpha=0.25)
            i += 1
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('neuron ID')
        ax.set_xlim(T[0], T[1])
        ax.set_ylim(neurons.min(), neurons.max())
        ax.set_title('database content on T = [%.0f, %.0f]' % (T[0], T[1]))


def test1():
    # need have a bunch of gdf files in current directory
    # delete old db
    os.system('rm test.db')

    # create db from excitatory files
    gdb = GDF('test.db', debug=True)
    gdb.create(re='brunel-py-ex-*.gdf', index=True)

    # get spikes for neurons 1,2,3
    spikes = gdb.select([1,2,3])

    # wont get any spikes for these neurons
    # cause they dont exist
    bad = gdb.select([100000,100001])

    gdb.close()

    print spikes
    print bad

if __name__ == '__main__':
    test1()
